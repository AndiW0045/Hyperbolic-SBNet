
import torch.nn as nn
import hyptorch.nn as hypnn

def make_fc_1d(f_in, f_out):
    return nn.Sequential(nn.Linear(f_in, f_out), 
                        nn.BatchNorm1d(f_out),
                        nn.ReLU(inplace=True),
                        nn.Dropout(p=0.5))


'''
Embedding Extraction Module
'''        

class EmbedBranch(nn.Module):
    def __init__(self, feat_dim, embedding_dim):
        super(EmbedBranch, self).__init__()
        self.fc1 = make_fc_1d(feat_dim, embedding_dim).cuda()
        self.fc2 = make_fc_1d(embedding_dim, embedding_dim)
        self.fc3 = make_fc_1d(embedding_dim, embedding_dim)

    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        #x = self.fc3(x)
        x = nn.functional.normalize(x) 
        return x

'''
Main Module
'''

class FOP(nn.Module):
    def __init__(self, args, feat_dim, n_class):
        super(FOP, self).__init__()
        
        self.embed_branch = EmbedBranch(feat_dim, args.dim_embed)

        self.c = 0.33
        self.ball_dim = 20
        self.ln = nn.Linear(args.dim_embed, self.ball_dim)
        self.tp = hypnn.ToPoincare(c=self.c, train_x=False, train_c =False, ball_dim=self.ball_dim)
        self.mlr = hypnn.HyperbolicMLR(ball_dim=self.ball_dim, n_classes=901, c=self.c)

        self.logits_layer = nn.Linear(args.dim_embed, n_class)

        if args.cuda:
            self.cuda()

    def forward(self, feats):
        feats = self.embed_branch(feats)
        #logits = self.logits_layer(feats) # switching out last layer for hyperbolic layers

        feats = self.ln(feats)
        feats = self.tp(feats)
        logits = self.mlr(feats, c=self.tp.c)
        
        return feats, logits
    
    def train_forward(self, feats):
        
        comb = self(feats)
        return comb
