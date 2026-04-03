import torch
import torch.nn as nn
import torch.nn.functional as F


def make_mlp(input_size, sizes,
            hidden_activation='ReLU',
            output_activation='ReLU',
            layer_norm=False):
    hidden_activation = getattr(nn, hidden_activation)
    if output_activation is not None:
        output_activation = getattr(nn, output_activation)
    layers = []
    n_layers = len(sizes)
    sizes = [input_size] + sizes
    for i in range(n_layers-1):
        layers.append(nn.Linear(sizes[i], sizes[i+1]))
        if layer_norm:
            layers.append(nn.LayerNorm(sizes[i+1])) 
        layers.append(hidden_activation())
    layers.append(nn.Linear(sizes[-2], sizes[-1]))
    if output_activation is not None:
        if layer_norm:
            layers.append(nn.LayerNorm(sizes[-1]))
        layers.append(output_activation())
    return nn.Sequential(*layers)


def train(model, train_loader, optimizer, device=None, scaler=None, weight=None):
    """Train loop with optional device/scaler/weight arguments.
    If any of these are not provided the function will use safe defaults so
    it does not depend on notebook-level globals.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if scaler is None:
        scaler = torch.amp.GradScaler(enabled=True)
    if weight is None:
        weight = 1

    true_positive = 0
    False_positive = 0
    False_negative = 0
    true_negative = 0
    total_loss = 0

    for batch in train_loader:
        optimizer.zero_grad()
        data = batch.to(device)
        with torch.amp.autocast(device_type="cuda", dtype=torch.float16, enabled=True):
            pred = model(data)
            loss = F.binary_cross_entropy_with_logits(pred.float(), data.y.float(), pos_weight=torch.tensor(weight))
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        total_loss += loss.item()
        true_positive += ((pred > 0.5) & (data.y ==1)).sum().item()
        true_negative += ((pred < 0.5) & (data.y ==0)).sum().item()
        False_negative += ((pred < 0.5) & (data.y==1)).sum().item()
        False_positive += ((pred > 0.5) & (data.y==0)).sum().item()
    acc = true_positive/(true_positive + False_positive + False_negative+1e-5)  

    return acc, total_loss


def evaluate(model, test_loader, device=None, weight=None):
    """Evaluation loop with optional device and weight.
    If not provided, safe defaults are used.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if weight is None:
        weight = 1

    true_positive = 0
    False_positive = 0
    False_negative = 0
    true_negative = 0
    total_loss = 0
    False_negative_thres = 0

    for batch in test_loader:
        data = batch.to(device)
        with torch.amp.autocast(device_type="cuda", dtype=torch.float16,enabled=True):
            pred = model(data)
            loss = F.binary_cross_entropy_with_logits(pred.float(), data.y.float(), pos_weight=torch.tensor(weight))
        total_loss += loss.item()
        true_positive += ((pred > 0.5) & (data.y ==1)).sum().item()
        true_negative += ((pred < 0.5) & (data.y ==0)).sum().item()
        False_negative += ((pred < 0.5) & (data.y==1)).sum().item()
        False_positive += ((pred > 0.5) & (data.y==0)).sum().item()
    acc = true_positive/(true_positive + False_positive + False_negative+1e-5)  

    return acc, total_loss, true_positive, true_negative, False_positive, False_negative


def make_mlp(input_size, sizes,
            hidden_activation='ReLU',
            output_activation='ReLU',
            layer_norm=False):
    hidden_activation = getattr(nn, hidden_activation)
    if output_activation is not None:
        output_activation = getattr(nn, output_activation)
    layers = []
    n_layers = len(sizes)
    sizes = [input_size] + sizes
    for i in range(n_layers-1):
        layers.append(nn.Linear(sizes[i], sizes[i+1]))
        if layer_norm:
            layers.append(nn.LayerNorm(sizes[i+1]))
        layers.append(hidden_activation())
    layers.append(nn.Linear(sizes[-2], sizes[-1]))
    if output_activation is not None:
        if layer_norm:
            layers.append(nn.LayerNorm(sizes[-1]))
        layers.append(output_activation())
    return nn.Sequential(*layers)


class InitialInputNetwork(nn.Module):

    def __init__(self, input_dim, hidden_dim,iteration, number_of_color=0, hidden_activation='ReLU',
                layer_norm=True):
        super(InitialInputNetwork, self).__init__()
        self.edge_network = make_mlp(3+number_of_color,
                                [hidden_dim]* iteration ,
                                hidden_activation=hidden_activation,
                                output_activation=None,
                                layer_norm=layer_norm)

    def forward(self, edge_corr):
        return self.edge_network(edge_corr).squeeze(-1)


class InitialEdgeNetwork(nn.Module):

    def __init__(self, input_dim, hidden_dim,iteration, hidden_activation='ReLU',
                layer_norm=True):
        super(InitialEdgeNetwork, self).__init__()
        self.edge_network = make_mlp(4,
                                [hidden_dim]* iteration ,
                                hidden_activation=hidden_activation,
                                output_activation=None,
                                layer_norm=layer_norm)

    def forward(self, edge_corr):
        return self.edge_network(edge_corr).squeeze(-1)


class InputEdgeNetwork(nn.Module):

    def __init__(self, input_dim, hidden_dim,iteration, hidden_activation='ReLU',
                layer_norm=True):
        super(InputEdgeNetwork, self).__init__()
        self.network = make_mlp(input_dim*2,
                                [hidden_dim]* iteration ,
                                hidden_activation=hidden_activation,
                                output_activation=None,
                                layer_norm=layer_norm)

    def forward(self, x, edge_index):
        start, end = edge_index
        edge_inputs = torch.cat([x[start], x[end]], dim=1)
        return self.network(edge_inputs).squeeze(-1)    


class NeighborNetwork(nn.Module):

    def __init__(self, input_dim, output_dim, iteration, hidden_activation='ReLU',
                layer_norm=True):
        super(NeighborNetwork, self).__init__()
        self.nodes_network = make_mlp(input_dim*2, 
                                [output_dim]*iteration,
                                hidden_activation=hidden_activation,
                                output_activation=hidden_activation,
                                layer_norm=layer_norm)
        self.attension_network = make_mlp(input_dim*2, 
                                [output_dim]*iteration,
                                hidden_activation=hidden_activation,
                                output_activation=hidden_activation,
                                layer_norm=layer_norm)

    def forward(self, x, e, edge_index):
        start, end = edge_index
        src = self.attension_network(torch.cat([e , x[start]], dim = 1))
        mi = torch.zeros(x.shape[0], src.size(1), device=src.device, dtype=src.dtype)
        index = end.view(-1, 1).expand(-1, src.size(1))
        mi = mi.scatter_add_(0, index, src)
        node_inputs = torch.cat([mi, x], dim=1)

        return self.nodes_network(node_inputs)    


class EdgeNetwork(nn.Module):

    def __init__(self, input_dim, hidden_dim,iteration, hidden_activation='ReLU',
                layer_norm=True):
        super(EdgeNetwork, self).__init__()
        self.network = make_mlp(input_dim*3,
                                [hidden_dim]* iteration ,
                                hidden_activation=hidden_activation,
                                output_activation=None,
                                layer_norm=layer_norm)

    def forward(self, x,e, edge_index):
        start, end = edge_index
        edge_inputs = torch.cat([x[start], x[end], e], dim=1)
        return self.network(edge_inputs).squeeze(-1)


class NodeNetwork(nn.Module):

    def __init__(self, input_dim, output_dim, iteration, hidden_activation='ReLU',
                layer_norm=True):
        super(NodeNetwork, self).__init__()
        self.nodes_network = make_mlp(input_dim*3, 
                                [output_dim]*iteration,
                                hidden_activation=hidden_activation,
                                output_activation=hidden_activation,
                                layer_norm=layer_norm)
        self.attension_network = make_mlp(input_dim*2, 
                                [output_dim]*iteration,
                                hidden_activation=hidden_activation,
                                output_activation=hidden_activation,
                                layer_norm=layer_norm)

    def forward(self, x, e, edge_index):
        start, end = edge_index
        src_mi = self.attension_network(torch.cat([e , x[start]], dim = 1))
        mi = torch.zeros(x.shape[0], src_mi.size(1), device=src_mi.device, dtype=src_mi.dtype)
        index_mi = end.view(-1, 1).expand(-1, src_mi.size(1))
        mi = mi.scatter_add_(0, index_mi, src_mi)
        src_mo = self.attension_network(torch.cat([e , x[end]], dim = 1))

        mo = torch.zeros(x.shape[0], src_mo.size(1), device=src_mo.device, dtype=src_mo.dtype)
        index_mo = start.view(-1, 1).expand(-1, src_mo.size(1))
        mo = mo.scatter_add_(0, index_mo, src_mo)
        node_inputs = torch.cat([mi, mo, x], dim=1)

        return self.nodes_network(node_inputs)


class FinalNetwork(nn.Module):

    def __init__(self, input_dim, hidden_dim,iteration, number_of_color=0, hidden_activation='ReLU',
                layer_norm=True):
        super(FinalNetwork, self).__init__()
        self.node_network = make_mlp(input_dim*7,
                                [hidden_dim]* iteration ,
                                hidden_activation=hidden_activation,
                                output_activation=None,
                                layer_norm=layer_norm)
        self.edge_network = make_mlp(input_dim*(4+number_of_color),
                                [hidden_dim]* iteration ,
                                hidden_activation=hidden_activation,
                                output_activation=None,
                                layer_norm=layer_norm)
        self.final_network = make_mlp(input_dim*3,
                                [hidden_dim]* 7 + [1],
                                hidden_activation=hidden_activation,
                                output_activation=None,
                                layer_norm=layer_norm)

    def forward(self, x, e, edge_index):
        x = self.node_network(x)
        e = self.edge_network(e)
        start, end = edge_index
        edge_inputs = torch.cat([x[start], x[end], e], dim=1)

        return self.final_network(edge_inputs).squeeze(-1)    


class AGNN_Network(nn.Module):

    def __init__(self, input_dim, hidden_dim, n_graph_iters,iteration, number_of_color=0, hidden_activation='ReLU', layer_norm=True):
        super(AGNN_Network, self).__init__()
        self.n_graph_iters = n_graph_iters
        self.input_node_network = make_mlp(input_dim, [hidden_dim] * 3,
                                    output_activation=hidden_activation,
                                    layer_norm=layer_norm)

        self.input_edge_network = InputEdgeNetwork(hidden_dim, hidden_dim, iteration,
                                        hidden_activation, layer_norm=layer_norm)
        self.edge_network = EdgeNetwork(hidden_dim, hidden_dim, iteration,
                                        hidden_activation, layer_norm=layer_norm)
        self.node_network = NodeNetwork(hidden_dim, hidden_dim, iteration,
                                        hidden_activation, layer_norm=layer_norm)
        self.final_network = FinalNetwork(hidden_dim, hidden_dim, iteration, number_of_color,
                                        hidden_activation, layer_norm=layer_norm)
        self.NeighborNetwork = NeighborNetwork(hidden_dim, hidden_dim, iteration,
                                        hidden_activation, layer_norm=layer_norm)

        self.initialedge_network = InitialEdgeNetwork(hidden_dim, hidden_dim, iteration,
                                        hidden_activation, layer_norm=layer_norm)

    def forward(self, inputs):
        x_initial = self.input_node_network(inputs.x)
        e_initial = self.initialedge_network(inputs.edge_corr)
        e_ne = torch.sigmoid(self.input_edge_network(x_initial, inputs.neighbor_edges))
        x_0 = self.NeighborNetwork(x_initial, e_ne, inputs.neighbor_edges)
        x_0 = x_0 + x_initial
        e_0 = torch.sigmoid(self.input_edge_network(x_0, inputs.edge_index))
        e_1 = torch.sigmoid(self.edge_network(x_0, e_0, inputs.edge_index))
        e_1 = e_1 + e_initial

        e_ne_1 = torch.sigmoid(self.input_edge_network(x_0, inputs.neighbor_edges))
        x_0 = self.NeighborNetwork(x_0, e_ne_1, inputs.neighbor_edges)
        x_1 = self.node_network(x_0, e_1, inputs.edge_index)
        e_2 = torch.sigmoid(self.edge_network(x_1, e_1, inputs.edge_index))

        e_ne_2 = torch.sigmoid(self.input_edge_network(x_1, inputs.neighbor_edges))
        x_1 = self.NeighborNetwork(x_1, e_ne_2, inputs.neighbor_edges)
        x_2 = self.node_network(x_1, e_2, inputs.edge_index)
        e_3 = torch.sigmoid(self.edge_network(x_2, e_2, inputs.edge_index))

        e_ne_3 = torch.sigmoid(self.input_edge_network(x_2, inputs.neighbor_edges))
        x_2 = self.NeighborNetwork(x_2, e_ne_3, inputs.neighbor_edges)
        x_3 = self.node_network(x_2, e_3, inputs.edge_index)
        e_4 = torch.sigmoid(self.edge_network(x_3, e_3, inputs.edge_index))

        e_ne_4 = torch.sigmoid(self.input_edge_network(x_3, inputs.neighbor_edges))
        x_3 = self.NeighborNetwork(x_3, e_ne_4, inputs.neighbor_edges)
        x_4 = self.node_network(x_3, e_4, inputs.edge_index)
        e_5 = torch.sigmoid(self.edge_network(x_4, e_4, inputs.edge_index))

        e_ne_5 = torch.sigmoid(self.input_edge_network(x_4, inputs.neighbor_edges))
        x_4 = self.NeighborNetwork(x_4, e_ne_5, inputs.neighbor_edges)
        x_5 = self.node_network(x_4, e_5, inputs.edge_index)
        x = torch.cat([x_initial, x_0, x_1, x_2, x_3, x_4, x_5], dim=1)
        e = torch.cat([e_initial, e_0, e_1, e_2, e_3, e_4, e_5], dim=1)

        return self.final_network(x,e, inputs.edge_index)
