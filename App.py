import streamlit as st
from rdkit import Chem
from rdkit.Chem import Draw
import torch
import joblib
import torch_geometric
from torch_geometric.data import Batch
import numpy as np
import pandas as pd
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, global_mean_pool, GraphNorm
import numpy as np
from minisom import MiniSom
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import random
import pickle



class GNNEncoder(nn.Module):
    def __init__(self, in_channels=9, hidden_channels=64):
        super().__init__()
        self.gat1 = GATConv(in_channels, hidden_channels)
        self.gat2 = GATConv(hidden_channels, hidden_channels)
        self.gat3 = GATConv(hidden_channels, hidden_channels)
        self.norm = GraphNorm(hidden_channels)
        self.dropout = nn.Dropout(0.25)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = F.leaky_relu(self.gat1(x, edge_index))
        x = self.norm(x, batch)
        x = self.dropout(x)
        x = F.leaky_relu(self.gat2(x, edge_index))
        x = x + self.gat3(x, edge_index)
        x = global_mean_pool(x, batch)
        return x

class SOMSoftLayer(nn.Module):
    def __init__(self, m, n, dim):
        super().__init__()
        self.m = m
        self.n = n
        self.dim = dim
        self.som = MiniSom(m, n, dim, sigma=0.7, learning_rate=0.3)
        self.initialized = False

    def forward(self, x):
        x_np = x.detach().cpu().numpy()
        if not self.initialized:
            self.som.random_weights_init(x_np)
            self.som.train_batch(x_np, 1000)
            self.initialized = True
        responses = np.array([
            self.som.activation_response([v])[0].flatten()
            for v in x_np
        ])
        return torch.from_numpy(responses).float().to(x.device)

class MultiPropertyPredictor(nn.Module):
    def __init__(self, num_properties=4):
        super().__init__()
        self.encoder = GNNEncoder()
        self.som = SOMSoftLayer(3, 3, 64)
        self.som_output_dim = None
        self.num_properties = num_properties
        self.regressor = None

    def forward(self, data):
        x = self.encoder(data)
        som_out = self.som(x)

        if self.som_output_dim is None:
            self.som_output_dim = som_out.shape[1]
            self.regressor = nn.Sequential(
                nn.Linear(64 + self.som_output_dim, 64),
                nn.ReLU(),
                nn.Linear(64, self.num_properties)
            )
            self.regressor.to(x.device)

        combined = torch.cat([x, som_out], dim=1)
        return self.regressor(combined)

# Define atom_features and smiles_to_graph exactly as in training
def atom_features(atom):
    return [
        atom.GetAtomicNum(),
        atom.GetTotalDegree(),
        atom.GetFormalCharge(),
        int(atom.GetHybridization()),
        int(atom.GetIsAromatic()),
        atom.GetTotalNumHs(includeNeighbors=True),
        atom.GetMass(),
        atom.GetExplicitValence(),
        atom.GetImplicitValence()
    ]

def smiles_to_graph(smiles):
    mol = Chem.MolFromSmiles(smiles)
    atoms = mol.GetAtoms()
    bonds = mol.GetBonds()
    x = torch.tensor([atom_features(a) for a in atoms], dtype=torch.float)
    edge_index = []
    for bond in bonds:
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        edge_index += [[i, j], [j, i]]
    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    return Batch.from_data_list([torch_geometric.data.Data(x=x, edge_index=edge_index)])

# === Model classes (copy from your model definition) ===
# Paste GNNEncoder, SOMSoftLayer, and MultiPropertyPredictor here...

# For brevity, we assume they are defined above this line in the file

# ==== Load trained model ====
property_names = ['alpha', 'HOMO', 'H', 'Cv']
scalers = [joblib.load(f"saved_model/scaler_{name}.pkl") for name in property_names]

model = MultiPropertyPredictor(num_properties=4)
dummy_batch = smiles_to_graph("CCO")
_ = model(dummy_batch)
model.load_state_dict(torch.load("saved_model/som_gnn_multitarget.pth"))
model.eval()

# ==== Streamlit UI ====
st.title("🧪 Molecular Property Predictor")
st.markdown("Enter a SMILES string to predict alpha, HOMO, enthalpy (H), and heat capacity (Cv) using a trained SOM + GNN model.")

smiles_input = st.text_input("🔤 Enter SMILES:", "CCO")

if st.button("Predict"):
    try:
        mol = Chem.MolFromSmiles(smiles_input)
        if mol is None:
            st.error("❌ Invalid SMILES string!")
        else:
            st.image(Draw.MolToImage(mol, size=(300, 300)), caption="Molecule Structure", use_column_width=False)

            graph = smiles_to_graph(smiles_input)
            with torch.no_grad():
                pred_scaled = model(graph)[0].cpu().numpy()

            predictions = {}
            for i, name in enumerate(property_names):
                pred = scalers[i].inverse_transform([[pred_scaled[i]]])[0][0]
                predictions[name] = pred

            st.subheader("📈 Predicted Properties:")
            for name, val in predictions.items():
                st.markdown(f"{name}: {val:.4f}")

    except Exception as e:
        st.error(f"An error occurred: {str(e)}")



st.markdown("---")
if st.button("📊 Evaluate on 100 Random Molecules"):
    try:
        # Load graphs and y_values (optional: cache these)
        import deepchem as dc
        tasks, datasets, _ = dc.molnet.load_qm9(featurizer='GraphConv')
        train_dataset, _, _ = datasets

        property_indices = [1, 2, 9, 11]
        smiles_list = []
        y_values = []

        for _, y, _, ids in train_dataset.iterbatches(batch_size=1, deterministic=True):
            smiles_list.append(ids[0])
            y_values.append(y[0][property_indices])

        y_values = np.array(y_values)
        graphs = [smiles_to_graph(s) for s in smiles_list]

        # Predict on 100 random molecules
        model.eval()
        random_indices = random.sample(range(len(graphs)), 100)
        predictions = {name: [] for name in property_names}
        actuals = {name: [] for name in property_names}

        with torch.no_grad():
            for i in random_indices:
                graph = graphs[i]
                batch_data = Batch.from_data_list([graph])
                pred_scaled = model(batch_data)[0].cpu().numpy()

                for j, name in enumerate(property_names):
                    actual = y_values[i, j]
                    pred = scalers[j].inverse_transform([[pred_scaled[j]]])[0][0]
                    actuals[name].append(actual)
                    predictions[name].append(pred)

        # Show metrics
        st.subheader("📊 Evaluation Metrics on 100 Random Molecules")
        metrics = []
        for name in property_names:
            mae = mean_absolute_error(actuals[name], predictions[name])
            rmse = mean_squared_error(actuals[name], predictions[name], squared=False)
            r2 = r2_score(actuals[name], predictions[name])
            metrics.append({
                "Property": name,
                "MAE": round(mae, 4),
                "RMSE": round(rmse, 4),
                "R² Score": round(r2, 4)
            })
        # Create and display the DataFrame
        metrics_df = pd.DataFrame(metrics)
        st.table(metrics_df)

        # Plot
        st.subheader("📈 Actual vs Predicted (100 Samples)")
        fig, axs = plt.subplots(2, 2, figsize=(20, 15))
        axs = axs.ravel()

        for i, name in enumerate(property_names):
            axs[i].plot(range(1, 101), actuals[name], label='Actual', marker='o', color='green')
            axs[i].plot(range(1, 101), predictions[name], label='Predicted', marker='s', color='blue')
            axs[i].set_xlabel("Molecule Index")
            axs[i].set_ylabel(f"{name} Value")
            axs[i].set_title(f"Actual vs Predicted: {name}")
            axs[i].legend()
            axs[i].grid(True)

        st.pyplot(fig)

    except Exception as e:
        st.error(f"An error occurred during evaluation: {str(e)}")

