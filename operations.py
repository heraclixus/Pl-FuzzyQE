import torch
import torch.nn as nn
import torch.nn.functional as F
from util import get_regularizer
from models import SimpleMLP


"""
Entity mapping: takes entity embeddings and map them into a PL-Fuzzy set in [0,1]^d
"""
class EntityMapping(nn.Module):
    def __init__(self, entity_dim, hidden_dim, 
                 num_hidden_layers,
                 regularizer,
                 n_partitions, modulelist):
        super(EntityMapping, self).__init__()
        self.entity_dim = entity_dim
        self.hidden_dim = hidden_dim
        self.num_hidden_layers = num_hidden_layers
        self.regularizer = regularizer
        self.n_partitions = n_partitions
        self.modulelist = modulelist
        
        # parallel einsum w.r.t partitions
        if self.modulelist:
            self.hidden_dim = self.hidden_dim
            self.mapping_weights1 = nn.Parameter(torch.ones((self.n_partitions, self.entity_dim, self.hidden_dim)))
            # (h, 1)
            self.mapping_weights2 = nn.Parameter(torch.ones((self.n_partitions, self.hidden_dim, 1)))
            self.relu = torch.relu
            self.sigmoid = torch.sigmoid
            nn.init.uniform_(tensor=self.mapping_weights1, a=0, b=1)
            nn.init.uniform_(tensor=self.mapping_weights2, a=0, b=1)
        else:
            self.pl_fuzzyset_maps = SimpleMLP(input_dim=self.entity_dim, hidden_dim=self.hidden_dim,
                                              num_hidden_layers=self.num_hidden_layers,
                                              regularizer=self.regularizer,
                                              output_dim=self.n_partitions) # n_partitions for output_dim 
    """
    e_embedding: either one embedding of shape (B,e)
    or negative samples with shape (B,n,e)
    returns shape (B,d) or (n,B,d)
    """
    def forward(self, e_embedding):
        # (B,e)
        # print(f"forward: e_embedding = {e_embedding.shape}")
        if len(e_embedding.shape) == 2:
            pl_fuzzyset = []
            if self.modulelist:
                e_embedding = e_embedding.unsqueeze(1).repeat(1,self.n_partitions, 1)        
                inter_embedding = self.relu(torch.einsum("bde,deh->bdh",e_embedding, self.mapping_weights1))
                return self.sigmoid(torch.einsum("bdh,dhl->bd", inter_embedding, self.mapping_weights2))
            else:
                return self.pl_fuzzyset_maps(e_embedding)
        else: # (B,n,e)
            pl_fuzzyset = []
            if self.modulelist:
                e_embedding = e_embedding.unsqueeze(1).repeat(1,self.n_partitions, 1, 1)
                inter_embedding = self.relu(torch.einsum("bdne,deh->bdnh", e_embedding, self.mapping_weights1))
                return self.sigmoid(torch.einsum("bdnh,dhl->bnd", inter_embedding, self.mapping_weights2)).permute(1,0,2)
            else:
                return self.pl_fuzzyset_maps(e_embedding).permute(1,0,2)
            
            

"""
For better Debugging purpose, write two classes
Projection MLP is used to perform projection using partitionwise-MLPs
"""
class ProjectionMLP(nn.Module):
    def __init__(
            self,
            nrelation,
            regularizer_setting,
            relation_dim,
            num_layers,
            projection_dim,
            n_partitions, # for plfuzzyset
            strict_partition, # for plfuzzy set
            modulelist,
    ):
        super(ProjectionMLP, self).__init__()
        self.regularizer = get_regularizer(regularizer_setting, n_partitions, neg_input_possible=True)
        self.relation_dim = relation_dim # TODO: should relation_dim = n_partitions? 
        self.strict_partition = strict_partition
        self.mlp_hidden_dim = projection_dim // n_partitions * 10  # TODO: one degree of freedom
        self.n_partitions = n_partitions
        self.num_layers = num_layers
        self.modulelist = modulelist
        # mlp
        # partition vs. non-partition
        self.relation_embedding = nn.Parameter(torch.zeros(nrelation, self.relation_dim))  # same dim
        nn.init.uniform_(tensor=self.relation_embedding, a=0, b=1)

        input_dim = 1 + self.relation_dim if self.strict_partition else self.n_partitions + self.relation_dim

        # one MLP for each partition
        # input as the concatenation of a plfuzzy set and r 
        # here the input_dim can be (a) n_partitions + relation_dim, or (b) 1 + relational_dim
        if self.modulelist:
            self.mapping_weights1 = nn.Parameter(torch.ones((self.n_partitions, input_dim, self.mlp_hidden_dim)))
            # (h, 1)
            self.mapping_weights2 = nn.Parameter(torch.ones((self.n_partitions, self.mlp_hidden_dim, 1)))
            self.relu = torch.relu
            self.sigmoid = torch.sigmoid
            nn.init.xavier_uniform_(self.mapping_weights1)
            nn.init.xavier_uniform_(self.mapping_weights2)
            
        else: # one MLP for all 
            self.MLPs = SimpleMLP(input_dim=input_dim, hidden_dim=projection_dim,
                                  num_hidden_layers=self.num_layers, regularizer=self.regularizer,
                                  output_dim=self.n_partitions) 

    # e_pl_fuzzyset = (B,1,d)
    # rid = list of rids 
    def forward(self, e_pl_fuzzyset, rid):
        r_embedding = torch.index_select(self.relation_embedding, dim=0, index=rid)
        if len(e_pl_fuzzyset.shape) == 3:
            e_pl_fuzzyset = e_pl_fuzzyset.squeeze(1)
        if self.modulelist:
            input = torch.cat([e_pl_fuzzyset, r_embedding], dim=-1) # (B,d+r)
            input_rep = input.unsqueeze(1).repeat(1,self.n_partitions, 1)   # (B,d,d+r)     
            inter_embedding = self.relu(torch.einsum("bdr,drh->bdh",input_rep, self.mapping_weights1))
            plfuzzy_set = self.sigmoid(torch.einsum("bdh,dhl->bd", inter_embedding, self.mapping_weights2))
        else:
            plfuzzy_set = self.MLPs(torch.cat([e_pl_fuzzyset, r_embedding],dim=-1))
        return plfuzzy_set


"""
The relational basis formulation for projection 
"""
class ProjectionRelBasis(nn.Module):
    def __init__(
            self,
            nrelation,
            regularizer_setting,
            relation_dim,
            projection_dim,
            n_partitions, # for plfuzzyset
    ):
        super(ProjectionRelBasis, self).__init__()
        self.regularizer = get_regularizer(regularizer_setting, n_partitions, neg_input_possible=True)
        self.relation_dim = relation_dim
        self.dual = regularizer_setting['dual']
        self.n_partitions = n_partitions
        self.hidden_dim = n_partitions  # TODO: degree of freedom
        
        # partition  vs. non-partition
        n_base = n_partitions // 10
        if not self.dual:
            self.rel_base = nn.Parameter(torch.zeros(n_base, self.hidden_dim, self.hidden_dim))
            self.rel_bias = nn.Parameter(torch.zeros(n_base, self.hidden_dim))
            self.rel_att = nn.Parameter(torch.zeros(nrelation, n_base)) # 474,500
            self.norm = nn.LayerNorm(self.hidden_dim, elementwise_affine=False)

            # new initialization
            torch.nn.init.orthogonal_(self.rel_base)
            torch.nn.init.xavier_normal_(self.rel_bias)
            torch.nn.init.xavier_normal_(self.rel_att)

        else:
            self.hidden_dim = self.hidden_dim // 2

            # for property vals
            self.rel_base1 = nn.Parameter(torch.randn(n_base, self.hidden_dim, self.hidden_dim))
            nn.init.uniform_(self.rel_base1, a=0, b=1e-2)
            self.rel_bias1 = nn.Parameter(torch.zeros(nrelation, self.hidden_dim))
            self.rel_att1 = nn.Parameter(torch.randn(nrelation, n_base))
            self.norm1 = nn.LayerNorm(self.hidden_dim, elementwise_affine=False)

            # for property weights
            self.rel_base2 = nn.Parameter(torch.randn(n_base, self.hidden_dim, self.hidden_dim))
            nn.init.uniform_(self.rel_base2, a=0, b=1e-2)
            self.rel_bias2 = nn.Parameter(torch.zeros(nrelation, self.hidden_dim))
            self.rel_att2 = nn.Parameter(torch.randn(nrelation, n_base))
            self.norm2 = nn.LayerNorm(self.hidden_dim, elementwise_affine=False)


    def forward(self, e_pl_fuzzyset, rid):
        if len(e_pl_fuzzyset.shape) == 3:
            e_pl_fuzzyset = e_pl_fuzzyset.squeeze(1)
        if not self.dual:
            project_r = torch.einsum('br,rio->bio', self.rel_att[rid], self.rel_base)
            if self.rel_bias.shape[0] == self.rel_base.shape[0]:
                bias = torch.einsum('br,ri->bi', self.rel_att[rid], self.rel_bias)
            else:
                bias = self.rel_bias[rid]
            output = torch.einsum('bio,bi->bo', project_r, e_pl_fuzzyset) + bias
            output = self.norm(output)
        else:
            e_embedding1, e_embedding2 = torch.chunk(e_pl_fuzzyset, 2, dim=-1)
            project_r1 = torch.einsum('br,rio->bio', self.rel_att1[rid], self.rel_base1)
            bias1 = self.rel_bias1[rid]
            output1 = torch.einsum('bio,bi->bo', project_r1, e_embedding1) + bias1
            output1 = self.norm1(output1)

            project_r2 = torch.einsum('br,rio->bio', self.rel_att2[rid], self.rel_base2)
            bias2 = self.rel_bias2[rid]
            output2 = torch.einsum('bio,bi->bo', project_r2, e_embedding2) + bias2
            output2 = self.norm1(output2)

            output = torch.cat((output1, output2), dim=-1)

        output = self.regularizer(output)
        return output



"""
a wrapper class around the above two projection types
"""
class Projection(nn.Module):
    def __init__(
            self,
            nrelation,
            regularizer_setting,
            relation_dim,
            projection_dim,
            num_layers,
            projection_type,
            n_partitions, # for plfuzzyset
            strict_partition, # for plfuzzy set
            modulelist,
    ):
        super(Projection, self).__init__()

      
        if projection_type == 'mlp':
            self.projection_net = ProjectionMLP(nrelation=nrelation, regularizer_setting=regularizer_setting, 
                                                relation_dim=relation_dim, num_layers=num_layers,
                                                projection_dim=projection_dim, n_partitions=n_partitions, 
                                                strict_partition=strict_partition, modulelist=modulelist)
        else:
            self.projection_net = ProjectionRelBasis(nrelation=nrelation, regularizer_setting=regularizer_setting, 
                                                     relation_dim=relation_dim, n_partitions=n_partitions, projection_dim=projection_dim)

    """
    e_pl_fuzzyset = (B,1,d) 
    squeeze it inside the subroutines 
    """
    def forward(self, e_pl_fuzzyset, rid):
        return self.projection_net(e_pl_fuzzyset, rid)



class Conjunction(nn.Module):
    def __init__(self, entity_dim, logic_type, regularizer_setting, use_attention='False', godel_gumbel_beta=0.01):
        super(Conjunction, self).__init__()
        self.logic = logic_type
        self.regularizer = get_regularizer(regularizer_setting, entity_dim)
        self.use_attention = use_attention
        self.entity_dim = entity_dim

        if logic_type == 'godel_gumbel':
            self.godel_gumbel_beta = godel_gumbel_beta
        if use_attention:
            self.conjunction_layer1 = nn.Linear(self.entity_dim, self.entity_dim)
            # self.conjunction_layer2 = nn.Linear(self.entity_dim, self.entity_dim)
            self.conjunction_layer2 = nn.Linear(self.entity_dim, 1)  # no dimension-wise attention
            nn.init.xavier_uniform_(self.conjunction_layer1.weight)
            nn.init.xavier_uniform_(self.conjunction_layer2.weight)
        self.norm = nn.LayerNorm(entity_dim, elementwise_affine=False)

    def forward(self, embeddings):
        """
        :param embeddings: shape (# of sets, batch, dim).
        :return embeddings: shape (batch, dim)
        """
        if self.logic == 'godel':
            if self.logic == 'godel':
                # conjunction(x,y) = min{x,y}
                embeddings, _ = torch.min(embeddings, dim=0)
            elif self.logic == 'godel_gumbel':
                # soft way to compute min
                embeddings = -self.godel_gumbel_beta * torch.logsumexp(
                    -embeddings / self.godel_gumbel_beta,
                    0
                )
            return embeddings
        else:  # logic == product
            if self.logic == 'luka':
                # conjunction(x,y) = max{0, x+y-1}
                embeddings = torch.sum(embeddings, dim=0) - embeddings.shape[0] + 1
            elif self.logic == 'product':
                if not self.use_attention:
                    # conjunction(x,y) = xy
                    embeddings = torch.prod(embeddings, dim=0)
                else:
                    attention = self.get_conjunction_attention(embeddings)
                    # attention conjunction(x,y) = (x^p)*(y^q), p+q=1
                    # compute in log scale
                    epsilon = 1e-7  # avoid torch.log(0)
                    embeddings = torch.log(embeddings+epsilon)
                    embeddings = torch.exp(torch.sum(embeddings * attention, dim=0))
            embeddings = self.norm(embeddings)
            return self.regularizer(embeddings)

    def get_conjunction_attention(self, embeddings):
        layer1_act = F.relu(self.conjunction_layer1(embeddings))  # (num_conj, batch_size, 2 * dim)
        attention = F.softmax(self.conjunction_layer2(layer1_act)/torch.sqrt(self.entity_dim), dim=0)  # (num_conj, batch_size, 1)
        return attention


class Disjunction(nn.Module):
    def __init__(self, entity_dim, logic_type, regularizer_setting, godel_gumbel_beta=0.01):
        super(Disjunction, self).__init__()
        self.logic = logic_type
        self.regularizer = get_regularizer(regularizer_setting, entity_dim)

        if logic_type == 'godel_gumbel':
            self.godel_gumbel_beta = godel_gumbel_beta

        self.norm = nn.LayerNorm(entity_dim, elementwise_affine=False)


    def forward(self, embeddings):
        """
        :param embeddings: shape (# of sets, batch, dim).
        :return embeddings: shape (batch, dim)
        """
        if self.logic == 'godel':
            if self.logic == 'godel':
                # disjunction(x,y) = max{x,y}
                embeddings, _ = torch.max(embeddings, dim=0)
                return embeddings
            elif self.logic == 'godel_gumbel':
                # soft way to compute max
                embeddings = self.godel_gumbel_beta * torch.logsumexp(
                    embeddings / self.godel_gumbel_beta,
                    0
                )
            return embeddings
        else:
            if self.logic == 'luka':
                # disjunction(x,y) = min{1, x+y}
                embeddings = torch.sum(embeddings, dim=0)
            else:  # self.logic == 'product'
                # disjunction(x,y) = x+y-xy
                embeddings = torch.sum(embeddings, dim=0) - torch.prod(embeddings, dim=0)
            return self.regularizer(embeddings)


class Negation(nn.Module):
    def __init__(self, entity_dim, logic_type, regularizer_setting):
        super(Negation, self).__init__()
        self.logic = logic_type
        self.regularizer = get_regularizer(regularizer_setting, entity_dim)

    def forward(self, embeddings):
        """
        :param embeddings: shape (# of sets, batch, dim).
        :return embeddings: shape (batch, dim)
        """
        # negation(x) = 1-x
        return 1 - embeddings


