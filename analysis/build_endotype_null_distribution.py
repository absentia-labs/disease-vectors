import numpy as np, pickle
from tqdm import tqdm
EMBEDDINGS_DIR = '/media/lleger/LaCie/mit/disease_vector/'

def zeroth_persistent_homology(distance_matrix):

    number_of_points = distance_matrix.shape[0]
    
    parent = np.arange(number_of_points)
    component_size = np.ones(number_of_points, dtype=int)
    results = np.zeros((number_of_points, 3))
    results[:,0], results[:,1], results[:,2] = 0.0, np.inf, 1.0
    merge_size_differences = []
    def find_root(vertex):
        while parent[vertex] != vertex:
            parent[vertex] = parent[parent[vertex]]
            vertex = parent[vertex]
        return vertex
    edges = [(distance_matrix[i, j], i, j) for i in range(number_of_points) for j in range(i+1, number_of_points)]
    edges.sort()
    for distance_value, i, j in edges:
        root_i, root_j = find_root(i), find_root(j)
        if root_i != root_j:
            merge_size_differences.append(abs(component_size[root_i] - component_size[root_j]))
            if component_size[root_i] < component_size[root_j]:
                root_i, root_j = root_j, root_i
            results[root_j, 1] = distance_value
            results[root_j, 2] = component_size[root_j]
            parent[root_j] = root_i
            component_size[root_i] += component_size[root_j]
    root = find_root(0)
    results[root, 2] = component_size[root]
    
    # persistent entropy
    finite = results[:,1] < np.inf
    lengths = results[finite,1] - results[finite,0]
    probabilities = lengths / lengths.sum()
    persistent_entropy = float((-np.sum(probabilities * np.log(probabilities))) / np.log(len(lengths)))
    print(persistent_entropy)
    return results, persistent_entropy

def precompute_null_unit(n_points=1000, n_iter=int(1e4), save_path=EMBEDDINGS_DIR + "endotyping_null_distribution.pkl"):
    deaths_unit, sizes = [], []
    for _ in tqdm(range(n_iter)):
        u = np.random.uniform(0, 1, size=(n_points, n_points))
        u = (u + u.T) / 2
        np.fill_diagonal(u, 0)
        results, _ = zeroth_persistent_homology(u)
        finite = results[:, 1] < np.inf
        deaths_unit.extend(results[finite, 1])
        sizes.extend(results[finite, 2])
    out = {
        n_points: {
            "deaths_unit": np.asarray(deaths_unit),
            "sizes": np.asarray(sizes)
        }
    }

    pickle.dump(out, open(save_path, "wb"))
    print(f"Saved null distribution for n={n_points} to {save_path}")
    return out

precompute_null_unit()