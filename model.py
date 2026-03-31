import math
import torch
from torch import nn
import torch.nn.functional as F
import vs_helper


class ScaledDotProductAttention(nn.Module):
    def __init__(self, d_k, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.sqrt_d_k = math.sqrt(d_k)

    def forward(self, Q, K, V):
        attn = torch.bmm(Q, K.transpose(2, 1))
        attn = attn / self.sqrt_d_k
        attn = torch.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        y = torch.bmm(attn, V)
        return y, attn


class MultiHeadAttention(nn.Module):
    def __init__(self, num_head=8, num_feature=1024, dropout=0.1):
        super().__init__()
        self.num_head = num_head
        self.num_feature = num_feature
        self.d_k = num_feature // num_head
        
        self.W_Q = nn.Linear(num_feature, num_feature)
        self.W_K = nn.Linear(num_feature, num_feature)
        self.W_V = nn.Linear(num_feature, num_feature)
        self.W_O = nn.Linear(num_feature, num_feature)
        
        self.attention = ScaledDotProductAttention(self.d_k, dropout)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(num_feature)

    def forward(self, x):
        batch_size, seq_len, _ = x.shape
        residual = x
        
        Q = self.W_Q(x).view(batch_size, seq_len, self.num_head, self.d_k).transpose(1, 2)
        K = self.W_K(x).view(batch_size, seq_len, self.num_head, self.d_k).transpose(1, 2)
        V = self.W_V(x).view(batch_size, seq_len, self.num_head, self.d_k).transpose(1, 2)
        
        Q = Q.contiguous().view(batch_size * self.num_head, seq_len, self.d_k)
        K = K.contiguous().view(batch_size * self.num_head, seq_len, self.d_k)
        V = V.contiguous().view(batch_size * self.num_head, seq_len, self.d_k)
        
        out, attn = self.attention(Q, K, V)
        
        out = out.view(batch_size, self.num_head, seq_len, self.d_k)
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, self.num_feature)
        out = self.W_O(out)
        out = self.dropout(out)
        
        out = self.layer_norm(out + residual)
        return out, attn


class CrossAttention(nn.Module):
    """Cross-attention between query video and support summary"""
    def __init__(self, num_head=8, num_feature=1024, dropout=0.1):
        super().__init__()
        self.num_head = num_head
        self.d_k = num_feature // num_head
        
        self.W_Q = nn.Linear(num_feature, num_feature)
        self.W_K = nn.Linear(num_feature, num_feature)
        self.W_V = nn.Linear(num_feature, num_feature)
        self.W_O = nn.Linear(num_feature, num_feature)
        
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(num_feature)
        self.sqrt_d_k = math.sqrt(self.d_k)

    def forward(self, query, key_value):
        batch_size, q_len, _ = query.shape
        _, kv_len, _ = key_value.shape
        residual = query
        
        Q = self.W_Q(query).view(batch_size, q_len, self.num_head, self.d_k).transpose(1, 2)
        K = self.W_K(key_value).view(batch_size, kv_len, self.num_head, self.d_k).transpose(1, 2)
        V = self.W_V(key_value).view(batch_size, kv_len, self.num_head, self.d_k).transpose(1, 2)
        
        attn = torch.matmul(Q, K.transpose(-2, -1)) / self.sqrt_d_k
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        
        out = torch.matmul(attn, V)
        out = out.transpose(1, 2).contiguous().view(batch_size, q_len, -1)
        out = self.W_O(out)
        out = self.dropout(out)
        
        return self.layer_norm(out + residual)


class AttentionExtractor(nn.Module):
    def __init__(self, num_head=8, num_feature=1024, dropout=0.1):
        super().__init__()
        self.mha = MultiHeadAttention(num_head, num_feature, dropout)

    def forward(self, x):
        out, _ = self.mha(x)
        return out


class TemporalConvBlock(nn.Module):
    def __init__(self, num_feature, dropout=0.1):
        super().__init__()
        self.conv1 = nn.Conv1d(num_feature, num_feature, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(num_feature, num_feature, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(num_feature)
        self.bn2 = nn.BatchNorm1d(num_feature)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(num_feature)

    def forward(self, x):
        residual = x
        x = x.transpose(1, 2)
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.dropout(x)
        x = self.bn2(self.conv2(x))
        x = x.transpose(1, 2)
        x = self.dropout(x)
        return self.layer_norm(x + residual)


class SEBlock(nn.Module):
    def __init__(self, num_feature, reduction=16):
        super().__init__()
        self.fc1 = nn.Linear(num_feature, num_feature // reduction)
        self.fc2 = nn.Linear(num_feature // reduction, num_feature)
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, s, c = x.shape
        squeeze = x.mean(dim=1)
        excitation = self.relu(self.fc1(squeeze))
        excitation = self.sigmoid(self.fc2(excitation))
        excitation = excitation.unsqueeze(1)
        return x * excitation


class GatedFusion(nn.Module):
    def __init__(self, num_feature):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(num_feature * 2, num_feature),
            nn.Sigmoid()
        )
        self.transform = nn.Linear(num_feature * 2, num_feature)
        self.layer_norm = nn.LayerNorm(num_feature)

    def forward(self, x1, x2):
        concat = torch.cat([x1, x2], dim=-1)
        gate = self.gate(concat)
        transformed = self.transform(concat)
        out = gate * x1 + (1 - gate) * transformed
        return self.layer_norm(out)


class Reconstruction(nn.Module):
    def __init__(self, num_feature):
        super().__init__()
        self.fc1 = nn.Linear(num_feature, num_feature)
        self.fc2 = nn.Linear(num_feature, num_feature)
        self.lrelu = nn.LeakyReLU(0.2, inplace=True)
        self.dropout = nn.Dropout(0.3)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, feature):
        hidden = self.lrelu(self.fc1(feature))
        hidden = self.dropout(hidden)
        out = self.fc2(hidden)
        return out


class STeMI(nn.Module):
    def __init__(self, num_feature, num_hidden, num_head, temporal_scales, spatial_scales, dropout=0.5):
        super().__init__()
        self.num_feature = num_feature
        self.num_hidden = num_hidden
        self.temporal_scales = temporal_scales
        self.spatial_scales = spatial_scales
        
        # Calculate spatial dimensions
        self.spatial_size = int(math.sqrt(num_feature))
        if self.spatial_size * self.spatial_size != num_feature:
            for i in range(int(math.sqrt(num_feature)), 0, -1):
                if num_feature % i == 0:
                    self.spatial_h = i
                    self.spatial_w = num_feature // i
                    break
        else:
            self.spatial_h = self.spatial_size
            self.spatial_w = self.spatial_size
        
        # Feature normalization (helps with training stability)
        self.input_norm = nn.LayerNorm(num_feature)
        
        # Core attention
        self.attention = AttentionExtractor(num_head, num_feature, dropout)
        
        # Cross-attention for query-support interaction
        self.cross_attention = CrossAttention(num_head, num_feature, dropout)
        
        # Feature enhancement
        self.se_block = SEBlock(num_feature, reduction=16)
        self.temporal_conv = TemporalConvBlock(num_feature, dropout)
        
        # Spatial processing
        self.spatial_fc_1 = nn.Linear(num_feature, num_feature)
        
        # Positional embeddings
        self.pos_embed_1 = nn.Parameter(torch.zeros(1, 1, num_feature))
        self.pos_embed_2 = nn.Parameter(torch.zeros(1, 1, num_feature))
        self.pos_embed_3 = nn.Parameter(torch.zeros(1, 1, num_feature))
        nn.init.trunc_normal_(self.pos_embed_1, std=.02)
        nn.init.trunc_normal_(self.pos_embed_2, std=.02)
        nn.init.trunc_normal_(self.pos_embed_3, std=.02)
        
        # Gated fusion
        self.gated_fusion = GatedFusion(num_feature)
        
        # Cross-attention fusion
        self.cross_fusion = GatedFusion(num_feature)
        
        # Reconstruction (lightweight)
        self.reconstruction = Reconstruction(num_feature)
        
        # Output layers
        self.layer_norm = nn.LayerNorm(num_feature)
        
        self.fc1 = nn.Sequential(
            nn.Linear(num_feature, num_hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.LayerNorm(num_hidden)
        )
        
        self.merge_extractor = nn.Sequential(
            nn.Linear(num_feature, num_feature),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.LayerNorm(num_feature)
        )
        
        # Prediction heads
        self.fc_cls = nn.Linear(num_hidden, 1)
        self.fc_loc = nn.Linear(num_hidden, 2)
        self.fc_ctr = nn.Linear(num_hidden, 1)
        
        # Learnable weights for multi-scale fusion
        self.temporal_weight = nn.Parameter(torch.ones(1))
        self.spatial_weight = nn.Parameter(torch.ones(1))
        self.cross_weight = nn.Parameter(torch.ones(1) * 0.5)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    def forward(self, x, support_feature, support_target):
        support_target = support_target.squeeze(0)
        support_summary = support_feature[:, support_target, :]
        
        # Input normalization
        x = self.input_norm(x)
        support_feature = self.input_norm(support_feature)
        support_summary = self.input_norm(support_summary)
        
        # Feature enhancement
        x_enhanced = self.se_block(x)
        x_enhanced = self.temporal_conv(x_enhanced)
        
        support_enhanced = self.se_block(support_feature)
        support_enhanced = self.temporal_conv(support_enhanced)
        
        # Cross-attention: query attends to support summary
        cross_features = self.cross_attention(x_enhanced, support_summary)
        
        # Spatial branch
        spatial_support_feature = support_enhanced.clone()
        spatial_support_feature = spatial_support_feature + self.pos_embed_1
        spatial_support_feature = self.spatial_fc_1(spatial_support_feature)
        
        spatial_support_summary = support_summary.clone()
        spatial_support_summary = spatial_support_summary + self.pos_embed_2
        spatial_support_summary = self.spatial_fc_1(spatial_support_summary)
        
        spatial_x = x_enhanced.clone()
        spatial_x = spatial_x + self.pos_embed_3
        spatial_x = self.spatial_fc_1(spatial_x)
        
        # Reshape for spatial processing
        support_feat_out = spatial_support_feature.view(
            spatial_support_feature.shape[0],
            spatial_support_feature.shape[1],
            self.spatial_h,
            self.spatial_w
        )
        support_summary_out = spatial_support_summary.view(
            spatial_support_summary.shape[0],
            spatial_support_summary.shape[1],
            self.spatial_h,
            self.spatial_w
        )
        x_out = spatial_x.view(
            spatial_x.shape[0],
            spatial_x.shape[1],
            self.spatial_h,
            self.spatial_w
        )
        
        recon_support = support_feat_out.clone()
        recon_x = x_out.clone()
        
        # Multi-scale spatial processing
        merge_scales_space = []
        height = x_out.shape[3]
        
        for i in range(self.spatial_scales):
            if i > 0:
                height = max(int(height / 2), 1)
                adapt_pool = nn.AdaptiveAvgPool2d((x_out.shape[2], height)).to(x.device)
                support_feat_out = adapt_pool(support_feat_out)
                support_summary_out = adapt_pool(support_summary_out)
                x_out = adapt_pool(x_out)
            
            merge_scale = torch.cat([support_feat_out, support_summary_out, x_out], 1)
            input_channels = merge_scale.shape[1]
            
            feature_compress = nn.Sequential(
                nn.Conv2d(input_channels, x_out.shape[1], kernel_size=1, bias=False),
                nn.ReLU(inplace=True),
            ).to(x.device)
            
            compress_merge = feature_compress(merge_scale)
            merge_scales_space.append(compress_merge)
        
        merge_scales_all_space = torch.cat(merge_scales_space, 3)
        merge_scales_all_space = F.interpolate(
            merge_scales_all_space,
            size=(merge_scales_all_space.shape[2], merge_scales_all_space.shape[2])
        )
        merge_scales_all_space = merge_scales_all_space.view(
            merge_scales_all_space.shape[0],
            merge_scales_all_space.shape[1],
            self.num_feature
        )
        
        # Temporal branch
        support_feat_out = self.attention(support_enhanced)
        support_feat_out = support_feat_out + support_enhanced
        
        support_summary_out = self.attention(support_summary)
        support_summary_out = support_summary_out + support_summary
        
        support_strengthen = torch.bmm(support_enhanced, support_summary.transpose(1, 2))
        _, _, dim = support_strengthen.shape
        fc_1 = nn.Linear(dim, self.num_feature).to(x.device)
        support_updim = fc_1(support_strengthen)
        
        x_out = self.attention(x_enhanced)
        x_out = x_out + x_enhanced
        
        # Multi-scale temporal processing
        merge_scales_tpl = []
        row_sfo = support_feat_out.shape[1]
        row_sso = support_summary_out.shape[1]
        row_sup = support_updim.shape[1]
        row_xot = x_out.shape[1]
        column = support_enhanced.shape[2]
        
        for i in range(self.temporal_scales):
            adapt_pool_sfo = nn.AdaptiveAvgPool2d((row_sfo, column)).to(x.device)
            adapt_pool_sso = nn.AdaptiveAvgPool2d((row_sso, column)).to(x.device)
            adapt_pool_sup = nn.AdaptiveAvgPool2d((row_sup, column)).to(x.device)
            adapt_pool_xot = nn.AdaptiveAvgPool2d((row_xot, column)).to(x.device)
            
            sfo_scale = adapt_pool_sfo(support_feat_out).unsqueeze(0)
            sso_scale = adapt_pool_sso(support_summary_out).unsqueeze(0)
            sup_scale = adapt_pool_sup(support_updim).unsqueeze(0)
            xot_scale = adapt_pool_xot(x_out).unsqueeze(0)
            
            merge_scale = torch.cat([sfo_scale, sso_scale, sup_scale, xot_scale], 2)
            merge_scale = F.interpolate(merge_scale, size=(x.shape[1], x.shape[2]))
            merge_scales_tpl.append(merge_scale)
            
            row_sfo = max(int(row_sfo / 2), 1)
            row_sso = max(int(row_sso / 2), 1)
            row_sup = max(int(row_sup / 2), 1)
            row_xot = max(int(row_xot / 2), 1)
        
        merge_scales_tpl = torch.stack(merge_scales_tpl, dim=2)
        merge_scales_all_tpl = torch.mean(merge_scales_tpl, 2).squeeze(0)
        
        # Weighted fusion
        weighted_temporal = self.temporal_weight * merge_scales_all_tpl
        weighted_spatial = self.spatial_weight * merge_scales_all_space
        weighted_cross = self.cross_weight * cross_features
        
        # Gated fusion (temporal + spatial)
        fused = self.gated_fusion(weighted_temporal, weighted_spatial)
        
        # Fuse with cross-attention features
        fused = self.cross_fusion(fused, weighted_cross)
        
        # Final processing
        merge_x = self.merge_extractor(fused)
        out = self.fc1(merge_x)
        
        _, seq_len, _ = x.shape
        pred_cls = self.fc_cls(out).sigmoid().view(seq_len)
        pred_loc = self.fc_loc(out).exp().view(seq_len, 2)
        pred_ctr = self.fc_ctr(out).sigmoid().view(seq_len)
        
        # Reconstruction
        recon_x = recon_x.view(recon_x.shape[0], recon_x.shape[1], self.num_feature)
        reconstruction_x = self.reconstruction(recon_x)
        
        recon_support = recon_support.view(recon_support.shape[0], recon_support.shape[1], self.num_feature)
        reconstruction_support = self.reconstruction(recon_support)
        
        return pred_cls, pred_loc, pred_ctr, reconstruction_x, reconstruction_support

    def predict(self, seq, support_seq, support_summary):
        pred_cls, pred_loc, pred_ctr, _, _ = self(seq, support_seq, support_summary)
        pred_cls *= pred_ctr
        pred_cls /= pred_cls.max() + 1e-8
        pred_bboxes = vs_helper.offset2bbox(pred_loc)
        return pred_cls, pred_bboxes
