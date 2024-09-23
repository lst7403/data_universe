import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from collections import deque
import bisect

__all__ = ["Node", "calculation", "graph_tree"]

def _k_means_model_with_best_sil_score(X, random_seed = 0, kmax = 10):
  highest_score = 0
  k_means_model = None

  for ks in range(2, kmax+1):
    kmeans = KMeans(n_clusters = ks, random_state=random_seed).fit(X)
    cur_score = silhouette_score(X, kmeans.labels_, metric = 'euclidean')
    if cur_score > highest_score:
      highest_score = cur_score
      k_means_model = kmeans

  return k_means_model

class Node:
    def __init__(self, id = None, center = None, index = None, up_lv = None, down_lv = []):
        self.id = id # unique id for node
        self.center = center # cluster center (vector)
        self.index = index # cashe of search result, first 100? result close to center (list of index)
        self.up_lv = up_lv
        self.down_lv = down_lv # sub-cluster (lsit of node)

    def info(self):
        print(self.id, "\n", self.center, "\n", self.up_lv.id, "\n", [n.id for n in self.down_lv])

class calculation:
    def l1(vec1, vec2):
        return np.sum(np.abs(np.array(vec1)-np.array(vec2)))

    def l2_efficient(vec1, vec2):
        return sum((np.array(vec1)-np.array(vec2))**2)
    
    def l2(vec1, vec2):
        return np.sqrt(sum((np.array(vec1)-np.array(vec2))**2))
    
    def reflection(vector, reflection_vector):
        return 2*reflection_vector - vector
    
    def translate_to_norm_vector(original_vector, max, min):
        return (original_vector - min)/(max - min)
    
    def cos_sim_and_ecul(vec1, vec2):
        # Calculate dot product between vec1 and vec2
        dot_product = np.dot(vec1, vec2)
        
        # Calculate magnitudes (norms) of vec1 and vec2
        norm_vec1 = np.linalg.norm(vec1)
        norm_vec2 = np.linalg.norm(vec2)

        cos = dot_product / (norm_vec1 * norm_vec2)
        ecul = np.sqrt(norm_vec1**2 + norm_vec2**2 - (2 * norm_vec1 * norm_vec2 * cos))
    
        return cos, ecul
    
    def calculate_special_coordinates_with_cos_sim(cos_sim, r):
        tem_rad = -np.arccos(cos_sim)
        if cos_sim >= 0:
            theta_rad = tem_rad+np.pi/4
        else:
            theta_rad = tem_rad-np.pi/4
            # Calculate x and y coordinates using NumPy's cos and sin functions
        x = r * np.cos(theta_rad)
        y = r * np.sin(theta_rad)
        
        return x, y

    def sort_dict():
        pass
             

# graph tree (cluster)
class graph_tree:
    def __init__(self, attr_name, X, X_maxmin):
        #store necessary val
        self.attr_name = attr_name
        self.X = X
        self.X_maxmin = X_maxmin

        self.nodes = []
        self.lv_start = []
        self.graphs = []

        # init para for storing
        self.recom = None # a list of node, [[attr 1 max node, attr 1 min node], [attr 2 max node, attr 2 min node]...]
        self.q_norm_vector = None # search query
        self.status = None # current status record of clusters
            

    ############################ tree building area
    def build_tree(self, min_threshold_for_clustering):

        # create root node
        root_center = np.mean(self.X, axis=0)
        init_idx = np.array([i for i in range(len(self.X))])
        root_node = Node(id=0, center=root_center, index=init_idx[np.argsort(np.sum(abs(self.X-root_center), axis=1))])
        self.nodes.append(root_node)
        self.lv_start = [0,1]

        recom_max = root_center.copy()
        recom_min = root_center.copy()
        recom = [[root_node, root_node] for _ in range(len(self.attr_name))]
        
        while True:
            start = self.lv_start[-2]
            end = self.lv_start[-1]

            counter_lv_node = 0

            new_graph = []

            for node_i in range(start, end):
                cur = self.nodes[node_i]
                # k means required parameter and train the model
                print(f"cur node {cur.id}")
                # print(f"no. of cur node data {len(cur.index)}")

                # kmeans part
                if len(cur.index) > min_threshold_for_clustering:
                    cur_X = self.X[cur.index] # get cur data by index
                    k_means_model = _k_means_model_with_best_sil_score(cur_X) # get kmeans model with best k
                    print(f"optimal K {k_means_model.n_clusters}")

                    # neccessary para for create node
                    # sorting the index
                    # cal the distance between each data to coresponding center
                    distance_to_self_center = [
                        calculation.l2(cur_X[i], k_means_model.cluster_centers_[k_means_model.labels_[i]]) 
                        for i in range(len(cur_X))
                    ]
                    
                    sorted_indix_from_dist = np.argsort(distance_to_self_center)
                    sorted_index = cur.index[sorted_indix_from_dist]
                    sorted_labels = k_means_model.labels_[sorted_indix_from_dist]

                    # create new nodes
                    new_node = [
                        Node(
                            id=i+end+counter_lv_node,
                            center=k_means_model.cluster_centers_[i],
                            index=sorted_index[np.where(sorted_labels == i)[0]],
                            up_lv=cur
                        )
                        for i in range(k_means_model.n_clusters)
                    ]
                    
                    self.nodes.extend(new_node)
                    
                    # make reference to down_lv
                    cur.down_lv = self.nodes[end+counter_lv_node:len(self.nodes)]
                    # print(f"{cur.id}'s child: {[n.id for n in cur.down_lv]} \n")

                    counter_lv_node += k_means_model.n_clusters

                else:
                    #add to the recom
                    for attr_i in range(len(cur.center)):
                        if cur.center[attr_i] >= recom_max[attr_i]:
                            recom_max[attr_i] = cur.center[attr_i]
                            recom[attr_i][0] = cur
                            # print(f"added to {self.attr_name[attr_i]} max")

                        elif cur.center[attr_i] <= recom_min[attr_i]:
                            recom_min[attr_i] = cur.center[attr_i]
                            recom[attr_i][1] = cur
                            # print(f"added to {self.attr_name[attr_i]} min")

                # graph part
                if node_i + 1 < end:
                    new_graph.append([calculation.l2(cur.center, self.nodes[after_node_i].center) for after_node_i in range(node_i+1,end)])

                # print(new_graph)
            
            self.graphs.append(new_graph)
            # print("all", self.graphs)

            if counter_lv_node == 0:
                break

            self.lv_start.append(len(self.nodes))
            # print(self.lv_start, "\n")
        
        self.recom = recom

        return 0

    ####################################### search funtion area
    def search(self, q_vector):
        q_norm_vector = calculation.translate_to_norm_vector(q_vector, max=self.X_maxmin[0], min=self.X_maxmin[1])
        self.q_norm_vector = q_norm_vector
        closest_node = self.nodes[0]
        queue = deque([self.nodes[0]])
        closest = calculation.l2(self.nodes[0].center, q_norm_vector)
        
        while queue:
            cur = queue.popleft()
            dist = calculation.l2(cur.center, q_norm_vector)
            # print(f"checking {cur.id}, dist: {dist}")
            
            if dist < closest:
                closest_node = cur
                closest = dist
                # print(f"cur node: {closest_node.id}, closest: {closest}")
                queue.extend(cur.down_lv)
            
            elif dist - calculation.l2(cur.center, self.X[cur.index[-1]]) < closest:
                # print(f"{cur.id} potential")
                queue.extend(cur.down_lv)
            else:
                continue
                # print(f"{cur.id} puned")

            # print([n.id for n in queue], "\n")

        self.cur = closest_node

        return closest_node
    
    def discovery_search(self, search_vector, node, lv_degree=3):
        reflection_node = node
        for lv in range(lv_degree):
            if reflection_node.up_lv:
                reflection_node = reflection_node.up_lv
            else:
                break
        print(f"reflection_node: {reflection_node.id} at lv up: {lv}")

        return self.get_sorted_node_within_lv(search_vector, reflection_node, lv=lv)
    
    # discovery_search_helper
    def get_sorted_node_within_lv(self, vector, reflection_node, lv=float("inf")):
        if lv < float("inf"):
            cos_sim = []
            nodes = []
            euclide_dist = []
            feature_vector = vector - reflection_node.center
            print("generate cos sim, ecul dist, and nodes sorted by cos sim")
        elif lv == float("inf"):
            sorted_node = []
            sorted_dist = []
            print("generate sorted list of leaf node for fast_knn")

        queue = deque([reflection_node])
        
        lv_counter = 0

        while lv_counter <= lv:
            if not queue:
                break

            # print([n.id for n in queue])
            size = len(queue)

            for _ in range(size):
                cur = queue.popleft()
                # print("poped", cur.id)
                if not(cur.down_lv) or lv_counter==lv:
                    # print("cal")

                    if lv < float("inf"):
                        cur_cos_sim_and_ecul = calculation.cos_sim_and_ecul(feature_vector, cur.center-reflection_node.center)

                        # cos_sim.append(cur_cos_sim_and_ecul[0])
                        # euclide_dist.append(cur_cos_sim_and_ecul[1])
                        # nodes.append(cur)

                        # below is sorted by cos sim
                        position = bisect.bisect_left(cos_sim, cur_cos_sim_and_ecul[0])
                        cos_sim.insert(position, cur_cos_sim_and_ecul[0])
                        euclide_dist.insert(position, cur_cos_sim_and_ecul[1])
                        nodes.insert(position, cur)

                    elif lv == float("inf"):
                        cur_dist = calculation.l2(vector, reflection_node.center)
                        position = bisect.bisect_left(sorted_dist, cur_dist)
                        sorted_dist.insert(position, cur_dist)
                        sorted_node.insert(position, cur)

                else:
                    queue.extend(cur.down_lv)

            lv_counter+=1

        if lv < float("inf"):
            # pop itself
            if nodes[-1].id == self.cur:
                cos_sim.pop()
                sorted_node.pop()
                euclide_dist.pop()
            return {"sorted_cos_sim": cos_sim, "euclide_dist": euclide_dist, "sorted_nodes": nodes}
        elif lv == float("inf"):
            return sorted_node
        
    def fast_knn(self, vector, node, k=10):
        sorted_node_to_vector = self.get_sorted_node_within_lv(vector, node)
        # print(f"finish sorted_node_to_vector: {[node.id for node in sorted_node_to_vector]}")
        sorted_knn_val = [float("inf") for i in range(k)]
        sorted_knn_res = [None for i in range(k)]

        for node in sorted_node_to_vector:
            # print(f"checking {node.id}")
            test = calculation.l2(node.center, vector) - calculation.l2(node.center, self.X[node.index[-1]])
            # print(f"potential: {test}, far of knn: {sorted_knn_val[-1]}")
            if calculation.l2(node.center, vector) - calculation.l2(node.center, self.X[node.index[-1]]) < sorted_knn_val[-1]:
                for index in node.index:
                    dist = calculation.l2(vector, self.X[index])
                    if dist < sorted_knn_val[-1]:
                        # print(f"add dist {dist} and data {index}")
                        sorted_knn_val.pop()
                        sorted_knn_res.pop()
                        position = bisect.bisect_left(sorted_knn_val, dist)
                        sorted_knn_val.insert(position, dist)
                        sorted_knn_res.insert(position, index)
                        # print("update", sorted_knn_val, "\n", sorted_knn_res)
            else:
                # print("early break of knn")
                break

        return sorted_knn_res
    
    def graph_search(self, node):
        belong_lv = bisect.bisect_right(self.lv_start, node.id)-1

        starting_node_id = self.lv_start[belong_lv]
        graph = self.graphs[belong_lv]
        # for row in graph:
        #     print(row)
        
        index_in_graph = node.id - starting_node_id

        nodes_id = np.array([id for id in range(starting_node_id, self.lv_start[belong_lv+1])])
        nodes_id = np.delete(nodes_id, index_in_graph)
        print("other nodes in this lv:", nodes_id)

        
        relations = [graph[row][index_in_graph-1-row] for row in range(index_in_graph)] if index_in_graph > 0 else []

        if index_in_graph < len(graph):
            relations.extend(graph[index_in_graph])
        print(relations)

        sorted_indices = np.argsort(relations)
        sorted_nodes_id = nodes_id[sorted_indices]
        print(sorted_nodes_id)

        sorted_node = [self.nodes[i] for i in sorted_nodes_id]
        
        return sorted_node

    
    ############################### check tree area
    def print_tree(self, node, prefix="", is_last=True):
        """ Recursively print the tree in a visual structure with connecting lines. """
        
        # Print the current node
        print(prefix + ("└── " if is_last else "├── ") + str(node.id))
        
        # Update the prefix for the next level (children)
        new_prefix = prefix + ("    " if is_last else "│   ")
        
        # Iterate over children
        if node.down_lv:
            for i, child in enumerate(node.down_lv):
                # Determine if the child is the last child
                is_last_child = (i == len(node.down_lv) - 1)
                # Recursively print the child nodes
                self.print_tree(child, new_prefix, is_last_child)

    
    ############################## get data area
    def get_midle_graph_data(self, node_id, lv_degree=3, query_vector=None):
        if query_vector is None:
            query_vector = self.nodes[node_id].center

        # init
        cur_cluster = self.nodes[node_id]

        graph = {
            "nodes": [{"id": node_id}],
            "links": []
        }

        neighbour_clusters = self.discovery_search(query_vector, cur_cluster, lv_degree=lv_degree)



        
        return neighbour_clusters


