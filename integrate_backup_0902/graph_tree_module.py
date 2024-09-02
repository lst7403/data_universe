import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

def k_means_model_with_best_sil_score(X, random_seed = 0, kmax = 10):
  highest_score = 0
  k_means_model = None

  for ks in range(2, kmax+1):
    kmeans = KMeans(n_clusters = ks, random_state=random_seed).fit(X)
    cur_score = silhouette_score(X, kmeans.labels_, metric = 'euclidean')
    if cur_score > highest_score:
      highest_score = cur_score
      k_means_model = kmeans

  return k_means_model

class node:
    def __init__(self, id = None, entity = None, center = None, index = None, up_lv = None, down_lv = None):
        self.id = id # unique id for node
        self.entity = entity # sorted center by value with column name
        self.center = center # cluster center (vector)
        self.index = index # cashe of search result, first 100? result close to center (list of index)
        self.up_lv = up_lv # super-cluster (node)
        self.down_lv = down_lv # sub-cluster (lsit of node)

class sim:
    def l1(vector_1, vector_2):
        return sum(abs(vector_1-vector_2))

    def l2(vector_1, vector_2):
        return sum((vector_1-vector_2)**2)
             

# graph tree (cluster)
class graph_tree:
    def __init__(self, attr_name, X, X_maxmin, min_cluster_size, searhing_query = None, recom = None):
        self.attr_name = attr_name
        self.X = X
        self.X_maxmin = X_maxmin
        self.min_cluster_size = min_cluster_size
        self.searhing_query = searhing_query
        self.recom = recom # a list of node, [[attr 1 max node, attr 1 min node], [attr 2 max node, attr 2 min node]...]
        
    
        # create root node
        init_center = np.mean(X, axis=0)
        init_idx = np.array([i for i in range(len(X))])
        self.root = node(id=0, entity=sorted(list(zip(self.attr_name, init_center))), center=init_center, index=init_idx[np.argsort(np.sum(abs(X-init_center), axis=1))])
        del (init_center, init_idx)
        print("root node is created")
    
    def search(self, query_vector):
        self.searhing_query = query_vector
        cur = self.root
        
        while cur.down_lv:
            cur_center_distance = sim.l2(query_vector, cur.center)
            # print(cur_center_distance)

            closest = cur_center_distance
            closest_node_position = -1

            for node in range(len(cur.down_lv)):
                if sim.l2(query_vector, cur.down_lv[node].center) < closest:
                    closest = sim.l2(query_vector, cur.down_lv[node].center)
                    closest_node_position = node

            if closest_node_position < 0:
                break
            
            cur = cur.down_lv[closest_node_position]

        return cur
        


    def build_tree(self):
        
        recom_max = self.root.center.copy()
        recom_min = self.root.center.copy()
        recom = [[self.root, self.root] for _ in range(len(self.attr_name))]
        print(recom_max, recom_min, recom)
        
        queue = []
        queue.append(self.root)
        
        counting_id = 1
        lv = 0

        while queue:
            size = len(queue)
            print("queue", size, queue)
            print(f"\n---------------------------------- level {lv} no of node {size} ----------------------------------------")
            for node_in_this_lv in range(size):
                print(f"\nhandling node {node_in_this_lv}, info:")
                # k means required parameter and train the model
                cur = queue.pop(0) # get cur node
                print(f"cur node {cur}")
                print(f"no. of cur node data {len(cur.index)}")
                if len(cur.index) > self.min_cluster_size:
                    cur_X = self.X[cur.index] # get cur data by index
                    k_means_model = k_means_model_with_best_sil_score(cur_X) # get kmeans model
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
                    new_down_lv = [node(id=i+counting_id, entity=sorted(list(zip(self.attr_name, k_means_model.cluster_centers_[i])), key=lambda item: item[1], reverse=True), center=k_means_model.cluster_centers_[i], up_lv=cur) for i in range(k_means_model.n_clusters)]
                    
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
                    
                    queue += new_down_lv
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