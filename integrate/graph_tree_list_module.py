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
    def __init__(self, id = None, entity = None, center = None, index = None, up_lv = None, down_lv = []):
        self.id = id # unique id for node
        self.center = center # cluster center (vector)
        self.index = index # cashe of search result, first 100? result close to center (list of index)
        self.down_lv = down_lv # sub-cluster (lsit of node)

class calculation:
    def l1(vector_1, vector_2):
        return np.sum(np.abs(np.array(vector_1)-np.array(vector_2)))

    def l2(vector_1, vector_2):
        return sum((np.array(vector_1)-np.array(vector_2))**2)
    
    def reflection(self, vector, reflection_vector):
        return 2*reflection_vector - vector
    
    def translate_to_norm_vector(original_vector, max, min):
        return (original_vector - min)/(max - min)
    
    def sort_dict():
        pass
             

# graph tree (cluster)
class graph_tree:
    def __init__(self, attr_name, X, X_maxmin):
        #store necessary val
        self.attr_name = attr_name
        self.X = X
        self.X_maxmin = X_maxmin

        self.parent = []
        self.nodes = []
        self.graph = []

        # init para for storing
        self.recom = None # a list of node, [[attr 1 max node, attr 1 min node], [attr 2 max node, attr 2 min node]...]
        self.q_norm_vector = None # search query
        self.status = None # current status record of clusters
            

    ############################ tree building area
    def build_tree(self, min_threshold_for_clustering):

        # create root node
        root_center = np.mean(self.X, axis=0)
        init_idx = np.array([i for i in range(len(self.X))])
        root_node = Node(id=0, entity=sorted(list(zip(self.attr_name, root_center))), center=root_center, index=init_idx[np.argsort(np.sum(abs(self.X-root_center), axis=1))])
        self.nodes.append(root_node)
        self.parent.append(-1)
        self.graph.append([])

        recom_max = root_center.copy()
        recom_min = root_center.copy()
        recom = [[root_node, root_node] for _ in range(len(self.attr_name))]
        
        counter_id = 1
        pointer = 0
        lv = 0

        while pointer < len(self.nodes):
            print(f"\n---------------------------------- level {lv} ----------------------------------------")
            cur = self.nodes[pointer]
            # k means required parameter and train the model
            print(f"cur node {cur.id}")
            print(f"no. of cur node data {len(cur.index)}")

            # kmeans part
            if len(cur.index) > min_threshold_for_clustering:
                cur_X = self.X[cur.index] # get cur data by index
                k_means_model = _k_means_model_with_best_sil_score(cur_X) # get kmeans model with best k
                print(f"optimal K {k_means_model.n_clusters}\n")

                # sorting the index
                #cal the distance between each data to coresponding center
                distance_to_self_center = [
                    calculation.l1(cur_X[i], k_means_model.cluster_centers_[k_means_model.labels_[i]]) 
                    for i in range(len(cur_X))
                ]
                
                sorted_indix_from_dist = np.argsort(distance_to_self_center)
                sorted_index = cur.index[sorted_indix_from_dist]
                sorted_labels = k_means_model.labels_[sorted_indix_from_dist]

                new_node = [Node(id=i+counter_id, center=k_means_model.cluster_centers_[i], index=sorted_index[np.where(sorted_labels == i)[0]]) for i in range(k_means_model.n_clusters)]

                cur.down_lv = [i.id for i in new_node]
                
                self.nodes.extend(new_node)
                self.parent.extend([cur.id] * k_means_model.n_clusters)
                self.graph.extend([self.graph[cur.id]+1] * k_means_model.n_clusters)
                # print(f"down lv of this node {new_down_lv}")
                print(self.parent,"\n",self.nodes,"\n",self.graph)

                counter_id += k_means_model.n_clusters

            else:
                #add to the recom
                for i in range(len(cur.center)):
                    if cur.center[i] >= recom_max[i]:
                        recom_max[i] = cur.center[i]
                        recom[i][0] = cur
                        print(f"added to {self.attr_name[i]} max")
                    elif cur.center[i] <= recom_min[i]:
                        recom_min[i] = cur.center[i]
                        recom[i][1] = cur
                        print(f"added to {self.attr_name[i]} min")
                

        pointer += 1
        
        self.recom = recom

        return 0

    ####################################### search funtion area
    def search(self, q_vector):
        q_norm_vector = calculation.translate_to_norm_vector(q_vector, max=self.X_maxmin[0], min=self.X_maxmin[1])
        self.q_norm_vector = q_norm_vector
        closest_node = self.root
        queue = deque([self.root])
        closest = calculation.l1(self.root.center, q_norm_vector)
        
        while queue:
            cur = queue.popleft()
            dist = calculation.l1(cur.center, q_norm_vector)
            # print(f"checking {cur.id}, dist: {dist}")
            
            if dist < closest:
                closest_node = cur
                closest = dist
                # print(f"cur node: {closest_node.id}, closest: {closest}")
                queue.extend(cur.down_lv)
            
            elif dist - calculation.l1(cur.center, self.X[cur.index[-1]]) < closest:
                # print(f"{cur.id} potential")
                queue.extend(cur.down_lv)
            else:
                continue
                # print(f"{cur.id} puned")

            # print([n.id for n in queue], "\n")

        self.cur = closest_node

        return closest_node
    
    def reflection_search(self, search_vector, node, degree, k=10):
        reflection_node = node
        for i in range(degree):
            if reflection_node.up_lv:
                reflection_node = reflection_node.up_lv
            else:
                break
        
        reflected_vector = calculation.reflection(search_vector, reflection_node.center)

        knn_res = self.fast_knn(reflected_vector, reflection_node, k=k)
        belong_node = self._get_closest_node_within_lv(reflected_vector, reflection_node, lv=i)

        return [belong_node, reflected_vector, knn_res]
    
    # reflection_search_helper
    def _get_closest_node_within_lv(self, reflected_vector, node, lv=float("inf")):
        if lv < float("inf"):
            sorted_node = node
            sorted_dist = float("inf")
            print("will return a one closest node within lv")
        elif lv == float("inf"):
            sorted_node = []
            sorted_dist = []
            print("will return a sorted list of leaf node")

        queue = deque([node])
        
        lv_counter = 0

        while lv_counter <= lv:
            print([n.id for n in queue])
            if not queue:
                break

            size = len(queue)
            for _ in range(size):
                cur = queue.popleft()
                print("poped", cur.id)
                if not(cur.down_lv) or lv_counter==lv:
                    print("cal", not(cur.down_lv), lv_counter==lv)
                    cur_dist = calculation.l1(reflected_vector, cur.center)

                    if lv < float("inf"):
                        if cur_dist < sorted_dist:
                            sorted_dist = cur_dist
                            sorted_node = cur

                    elif lv == float("inf"):
                        position = bisect.bisect_left(sorted_dist, cur_dist)
                        sorted_dist.insert(position, cur_dist)
                        sorted_node.insert(position, cur)

                else:
                    queue.extend(cur.down_lv)

            lv_counter+=1

        return sorted_node
        
    def fast_knn(self, vector, node, k=10):
        sorted_node_to_vector = self._get_closest_node_within_lv(vector, node, mode="leaf")
        # print(f"finish sorted_node_to_vector: {[node.id for node in sorted_node_to_vector]}")
        sorted_knn_val = [float("inf") for i in range(k)]
        sorted_knn_res = [None for i in range(k)]

        for node in sorted_node_to_vector:
            # print(f"checking {node.id}")
            test = calculation.l1(node.center, vector) - calculation.l1(node.center, self.X[node.index[-1]])
            # print(f"potential: {test}, far of knn: {sorted_knn_val[-1]}")
            if calculation.l1(node.center, vector) - calculation.l1(node.center, self.X[node.index[-1]]) < sorted_knn_val[-1]:
                for index in node.index:
                    dist = calculation.l1(vector, self.X[index])
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
    def get_suround_cluster(self, node, query_vector=None):
        if query_vector is None:
            query_vector = node.center

        super_cluster = node.up_lv if node.up_lv else None

        tmp = node.up_lv if node.up_lv else None
        neibour_cluster = [n for n in tmp.down_lv if n != node] if tmp else None

        sub_cluster = node.down_lv if node.down_lv else None

        c1 = self.reflection_search(query_vector, node, degree=1)
        c2 = self.reflection_search(query_vector, node, degree=2)
        contra_cluster = [
            Node(id=-1, entity=calculation.reflection(query_vector, c1[0].center), center=c1[0], index=c1[1]),
            Node(id=-2, entity=calculation.reflection(query_vector, c2[0].center), center=c2[0], index=c2[1])
        ]
        res = {
            "cur_cluster": node,
            "super_cluster": super_cluster,
            "neibour_cluster": neibour_cluster,
            "sub_cluster": sub_cluster,
            "contra_cluster": contra_cluster
        }

        self.status = res
        
        return res


