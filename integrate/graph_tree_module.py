import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from collections import deque
import bisect

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
    def __init__(self, id = None, entity = None, center = None, index = None, up_lv = None, down_lv = None):
        self.id = id # unique id for node
        self.entity = entity # sorted center by value with column name
        self.center = center # cluster center (vector)
        self.index = index # cashe of search result, first 100? result close to center (list of index)
        self.up_lv = up_lv # super-cluster (node)
        self.down_lv = down_lv # sub-cluster (lsit of node)

class sim:
    def l1(vector_1, vector_2):
        return sum(abs(np.array(vector_1)-np.array(vector_2)))

    def l2(vector_1, vector_2):
        return sum((np.array(vector_1)-np.array(vector_2))**2)
             

# graph tree (cluster)
class graph_tree:
    def __init__(self, attr_name, X, X_maxmin, min_cluster_size):
        self.attr_name = attr_name
        self.X = X
        self.X_maxmin = X_maxmin
        self.min_cluster_size = min_cluster_size
        self.recom = None # a list of node, [[attr 1 max node, attr 1 min node], [attr 2 max node, attr 2 min node]...]
        self.q_vector = None # search query
        self.status = None # current status record of clusters
            
        # create root node
        init_center = np.mean(X, axis=0)
        init_idx = np.array([i for i in range(len(X))])
        self.root = Node(id=0, entity=sorted(list(zip(self.attr_name, init_center))), center=init_center, index=init_idx[np.argsort(np.sum(abs(X-init_center), axis=1))])
        del (init_center, init_idx)
        print("root node is created")
    
    def translate_to_norm_vector(self, original_vector):
        return (original_vector - self.X_maxmin[1])/(self.X_maxmin[0] - self.X_maxmin[1])

    def search(self, q_vector):
        q_vector = self.translate_to_norm_vector(q_vector)
        self.q_vector = q_vector
        closest_node = self.root
        queue = deque([self.root])
        closest = sim.l1(self.root.center, q_vector)
        
        while queue:
            cur = queue.popleft()
            dist = sim.l1(cur.center, q_vector)
            print(f"checking {cur.id}, dist: {dist}")
            
            if dist < closest:
                closest_node = cur
                closest = dist
                print(f"cur node: {closest_node.id}, closest: {closest}")
                queue.extend(cur.down_lv if cur.down_lv else [])
            
            elif dist - sim.l1(cur.center, self.X[cur.index[-1]]) < closest:
                print(f"{cur.id} potential")
                queue.extend(cur.down_lv if cur.down_lv else [])
            else:
                print(f"{cur.id} puned")

            print([n.id for n in queue], "\n")

        self.get_suround_cluster(closest_node, q_vector)

        return closest_node
        

    def build_tree(self):
        recom_max = self.root.center.copy()
        recom_min = self.root.center.copy()
        recom = [[self.root, self.root] for _ in range(len(self.attr_name))]
        print(recom_max, recom_min, recom)
        
        queue = deque([self.root])
        
        counting_id = 1
        lv = 0

        while queue:
            size = len(queue)
            print("queue", size, queue)
            print(f"\n---------------------------------- level {lv} no of node {size} ----------------------------------------")
            for node_in_this_lv in range(size):
                print(f"\nhandling node {node_in_this_lv}, info:")
                # k means required parameter and train the model
                cur = queue.popleft() # get cur node
                print(f"cur node {cur}")
                print(f"no. of cur node data {len(cur.index)}")
                if len(cur.index) > self.min_cluster_size:
                    cur_X = self.X[cur.index] # get cur data by index
                    k_means_model = _k_means_model_with_best_sil_score(cur_X) # get kmeans model
                    print(f"optimal K {k_means_model.n_clusters}\n")

                    # # relationship
                    # attr_no = 3
                    # largest_indices = np.argpartition(np.std(k_means_model.cluster_centers_, axis=0), -attr_no)[-attr_no:]
                    # print(largest_indices)

                    # relationship = {}
                    # np.argpartition(np.std(k_means_model.cluster_centers_, axis=0), -attr_no)[-attr_no:]
                    # for i in largest_indices:
                    #     tmp_entity[self.attr_name[i]]=k_means_model.cluster_centers_[groups][i]

                    # create required node
                    new_down_lv = [Node(id=i+counting_id, entity=sorted(list(zip(self.attr_name, k_means_model.cluster_centers_[i])), key=lambda item: item[1], reverse=True), center=k_means_model.cluster_centers_[i], up_lv=cur) for i in range(k_means_model.n_clusters)]
                    
                    # sorting the index
                    #cal the distance between each data to coresponding center
                    distance_to_self_center = []
                    for i in range(len(cur_X)):
                        distance_to_self_center.append(sim.l1(cur_X[i], k_means_model.cluster_centers_[k_means_model.labels_[i]]))
                    
                    sorted_indices = np.argsort(np.array(distance_to_self_center))
                    sorted_index = cur.index[sorted_indices]
                    sorted_labels = k_means_model.labels_[sorted_indices]

                    for i in range(k_means_model.n_clusters):
                        new_down_lv[i].index = sorted_index[np.where(sorted_labels == i)[0]]

                    new_down_lv.sort(key=lambda x: sim.l1(x.center, cur.center))
                    cur.down_lv = new_down_lv
                    
                    queue.extend(new_down_lv)
                    # print(f"down lv of this node {new_down_lv}")

                    counting_id += k_means_model.n_clusters
                else:
                    for i in range(len(cur.center)):
                        if cur.center[i] >= recom_max[i]:
                            recom_max[i] = cur.center[i]
                            recom[i][0] = cur
                            print(f"in {self.attr_name[i]} max")
                            print(recom)
                        elif cur.center[i] <= recom_min[i]:
                            recom_min[i] = cur.center[i]
                            recom[i][1] = cur
                            print(f"in {self.attr_name[i]} min")
                            print(recom)
                    

            lv += 1
            
        self.recom = recom

        return 0
    
    def print_tree(self, simple=1):
        queue = [self.root]
        lv = 0
        while queue:
            print(f"---------- lv: {lv}, node num.: {len(queue)} ----------")
            new_queue = []
            for n in queue:
                if simple == 0:
                    print(n.id)
                if n.down_lv:
                    new_queue += n.down_lv
            queue = new_queue
            lv += 1
    
    def _reflection(self, vector, reflection_vector):
        print(vector, reflection_vector)
        return 2*reflection_vector - vector
    
    def _get_sorted_leaf_node_dfs(self, vector, node):
        queue = deque([node])
        sorted_val = []
        sorted_leaf_nodes = []
        
        while queue:
            node = queue.popleft()
            dist = sim.l1(node.center, vector)
            # print(f"checking {node.id}, dist: {dist}")
            
            # Check if it's a leaf node
            if node.down_lv:
                queue.extend(node.down_lv)
                # print("queue: ", [n.id for n in queue])
            else:
                position = bisect.bisect_left(sorted_val, dist)
                sorted_val.insert(position, dist)
                sorted_leaf_nodes.insert(position, node)
                # print("sorted node: ",sorted_val,"\n", [n.id for n in sorted_leaf_nodes])

        return sorted_leaf_nodes
        

    def _fast_knn(self, vector, node, k=10):
        sorted_node_to_vector = self._get_sorted_leaf_node_dfs(vector, node)
        # print(f"finish sorted_node_to_vector: {[node.id for node in sorted_node_to_vector]}")
        sorted_knn_val = [float("inf") for i in range(k)]
        sorted_knn_res = [None for i in range(k)]

        for node in sorted_node_to_vector:
            # print(f"checking {node.id}")
            test = sim.l1(node.center, vector) - sim.l1(node.center, self.X[node.index[-1]])
            # print(f"potential: {test}, far of knn: {sorted_knn_val[-1]}")
            if sim.l1(node.center, vector) - sim.l1(node.center, self.X[node.index[-1]]) < sorted_knn_val[-1]:
                for index in node.index:
                    dist = sim.l1(vector, self.X[index])
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
    
    def reflection_search(self, search_vector, node, degree=1, k=10):
        reflection_node = node
        for i in range(degree):
            if reflection_node.up_lv:
                reflection_node = reflection_node.up_lv
            else:
                break
        
        reflected_vector = self._reflection(search_vector, reflection_node.center)
        return [reflection_node, self._fast_knn(reflected_vector, reflection_node, k=k)]

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
            Node(id=-1, entity=self._reflection(query_vector, c1[0].center), center=c1[0], index=c1[1]),
            Node(id=-2, entity=self._reflection(query_vector, c2[0].center), center=c2[0], index=c2[1])
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
