import torch
from torch.distributions import Categorical

class ACO():

    def __init__(self, 
                 distances,
                 n_ants=20, 
                 decay=0.9,
                 alpha=1,
                 beta=1,
                 elitist=False,
                 min_max=False,
                 pheromone=None,
                 heuristic=None,
                 min=None,
                 device='cpu'
                 ):
        
        self.problem_size = len(distances)
        self.distances  = distances
        self.n_ants = n_ants
        self.decay = decay
        self.alpha = alpha
        self.beta = beta
        self.elitist = elitist
        self.min_max = min_max
        
        # Dòng này đã được sửa đúng từ lần trước
        self.lowest_cost_path = None 
                     
        if min_max:
            if min is not None:
                assert min > 1e-9
            else:
                min = 0.1
            self.min = min
            self.max = None
        
        if pheromone is None:
            self.pheromone = torch.ones_like(self.distances)
            if min_max:
                self.pheromone = self.pheromone * self.min
        else:
            self.pheromone = pheromone

        self.heuristic = 1 / distances if heuristic is None else heuristic

        self.shortest_path = None
        self.lowest_cost = float('inf')

        self.device = device

    @torch.no_grad()
    def sparsify(self, k_sparse):
        '''
        Sparsify the TSP graph to obtain the heuristic information 
        Used for vanilla ACO baselines
        '''
        _, topk_indices = torch.topk(self.distances, 
                                        k=k_sparse, 
                                        dim=1, largest=False)
        edge_index_u = torch.repeat_interleave(
            torch.arange(len(self.distances), device=self.device),
            repeats=k_sparse
            )
        edge_index_v = torch.flatten(topk_indices)
        sparse_distances = torch.ones_like(self.distances) * 1e10
        sparse_distances[edge_index_u, edge_index_v] = self.distances[edge_index_u, edge_index_v]
        self.heuristic = 1 / sparse_distances
    
    def sample(self):
        paths, log_probs = self.gen_path(require_prob=True)
        costs = self.gen_path_costs(paths)
        return costs, log_probs

    # === BẮT ĐẦU SỬA HÀM RUN ===
    @torch.no_grad()
    def run(self, n_iterations, return_history=False): # <<< SỬA 1: Thêm return_history
        all_time_best_costs = [] # <<< SỬA 2: Thêm list rỗng
        
        for _ in range(n_iterations):
            paths = self.gen_path(require_prob=False)
            costs = self.gen_path_costs(paths)
            
            best_cost, best_idx = costs.min(dim=0)
            if best_cost < self.lowest_cost:
                self.shortest_path = paths[:, best_idx]
                self.lowest_cost = best_cost
                
                # Dòng này đã được sửa đúng từ lần trước
                self.lowest_cost_path = paths[:, best_idx] 

                if self.min_max:
                    max = self.problem_size / self.lowest_cost
                    if self.max is None:
                        self.pheromone *= max/self.pheromone.max()
                    self.max = max
            
            # <<< SỬA 3: Thêm khối 'if' này >>>
            if return_history:
                # Lưu lại 'lowest_cost' ở *mỗi* vòng lặp
                # (Dùng .item() để lấy giá trị Python thuần túy)
                all_time_best_costs.append(self.lowest_cost.cpu().item()) 
            
            self.update_pheronome(paths, costs)

        # <<< SỬA 4: Sửa lệnh 'return' >>>
        if return_history:
            return self.lowest_cost, all_time_best_costs
        else:
            return self.lowest_cost # Trả về như cũ
        # === KẾT THÚC SỬA HÀM RUN ===
       
    @torch.no_grad()
    def update_pheronome(self, paths, costs):
        '''
        Args:
            paths: torch tensor with shape (problem_size, n_ants)
            costs: torch tensor with shape (n_ants,)
        '''
        self.pheromone = self.pheromone * self.decay 
        
        if self.elitist:
            best_cost, best_idx = costs.min(dim=0)
            best_tour= paths[:, best_idx]
            self.pheromone[best_tour, torch.roll(best_tour, shifts=1)] += 1.0/best_cost
            self.pheromone[torch.roll(best_tour, shifts=1), best_tour] += 1.0/best_cost
        
        else:
            for i in range(self.n_ants):
                path = paths[:, i]
                cost = costs[i]
                self.pheromone[path, torch.roll(path, shifts=1)] += 1.0/cost
                self.pheromone[torch.roll(path, shifts=1), path] += 1.0/cost
                
        if self.min_max:
            self.pheromone[(self.pheromone > 1e-9) * (self.pheromone) < self.min] = self.min
            self.pheromone[self.pheromone > self.max] = self.max
    
    @torch.no_grad()
    def gen_path_costs(self, paths):
        '''
        Args:
            paths: torch tensor with shape (problem_size, n_ants)
        Returns:
                Lengths of paths: torch tensor with shape (n_ants,)
        '''
        assert paths.shape == (self.problem_size, self.n_ants)
        u = paths.T # shape: (n_ants, problem_size)
        v = torch.roll(u, shifts=1, dims=1)  # shape: (n_ants, problem_size)
        assert (self.distances[u, v] > 0).all()
        return torch.sum(self.distances[u, v], dim=1)

    def gen_path(self, require_prob=False):
        '''
        Tour contruction for all ants
        Returns:
            paths: torch tensor with shape (problem_size, n_ants), paths[:, i] is the constructed tour of the ith ant
            log_probs: torch tensor with shape (problem_size, n_ants), log_probs[i, j] is the log_prob of the ith action of the jth ant
        '''
        start = torch.randint(low=0, high=self.problem_size, size=(self.n_ants,), device=self.device)
        mask = torch.ones(size=(self.n_ants, self.problem_size), device=self.device)
        mask[torch.arange(self.n_ants, device=self.device), start] = 0
        
        paths_list = [] # paths_list[i] is the ith move (tensor) for all ants
        paths_list.append(start)
        
        log_probs_list = [] # log_probs_list[i] is the ith log_prob (tensor) for all ants' actions
        
        prev = start
        for _ in range(self.problem_size-1):
            actions, log_probs = self.pick_move(prev, mask, require_prob)
            paths_list.append(actions)
            if require_prob:
                log_probs_list.append(log_probs)
                mask = mask.clone()
            prev = actions
            mask[torch.arange(self.n_ants, device=self.device), actions] = 0
        
        if require_prob:
            return torch.stack(paths_list), torch.stack(log_probs_list)
        else:
            return torch.stack(paths_list)
        
    def pick_move(self, prev, mask, require_prob):
        '''
        Args:
            prev: tensor with shape (n_ants,), previous nodes for all ants
            mask: bool tensor with shape (n_ants, p_size), masks (0) for the visited cities
        '''
        pheromone = self.pheromone[prev] # shape: (n_ants, p_size)
        heuristic = self.heuristic[prev] # shape: (n_ants, p_size)
        dist = ((pheromone ** self.alpha) * (heuristic ** self.beta) * mask) # shape: (n_ants, p_size)
        dist = Categorical(dist)
        actions = dist.sample() # shape: (n_ants,)
        log_probs = dist.log_prob(actions) if require_prob else None # shape: (n_ants,)
        return actions, log_probs
        


if __name__ == '__main__':
    torch.set_printoptions(precision=3,sci_mode=False)
    input = torch.rand(size=(5, 2))
    distances = torch.norm(input[:, None] - input, dim=2, p=2)
    distances[torch.arange(len(distances)), torch.arange(len(distances))] = 1e10
    aco = ACO(distances)
    aco.sparsify(k_sparse=3)
    # Sửa lại lệnh test, truyền return_history=False (mặc định)
    print(aco.run(20, return_history=False))

