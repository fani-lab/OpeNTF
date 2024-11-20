try:
    from torch_geometric.data import Data, HeteroData
    from torch_scatter import scatter
    from torch_sparse import SparseTensor
    from torch_cluster import knn_graph
    from torch_spline_conv import spline_conv
    print("All necessary packages imported successfully!")
except ImportError as e:
    print(f"Error importing packages: {e}")
