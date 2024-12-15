import numpy as np

#透明化了卫星位置，简化了吞吐量计算
class LEOSatEnvironment:
    def __init__(self, config):
        self.config = config
        self.current_step = 0
        self.sat_features = None #每个平面最近卫星的位置
        self.prev_actions = None #每个UT之前的动作
        self.throughput = None #每个UT当前的吞吐量
        self.collision_info = None #每个UT当前是否碰撞
        self.reset()

    def reset(self):
        self.current_step = 0
        self.sat_features = np.random.rand(self.config.K)
        self.prev_actions = np.zeros(self.config.J, dtype=int)
        self.throughput = np.zeros(self.config.J)
        self.collision_info = np.zeros(self.config.J)
        return self._get_states()
    
    def step(self, chosen_planes): #chosen_planes为action，表示每个UT选择的动作
        preambles = self._assign_preambles(chosen_planes)
        plane_preamble_map = self._build_plane_preamble_map(chosen_planes, preambles)
        R_j, c_j = self._compute_throughput_and_collisions(plane_preamble_map)
        
        rewards = self._compute_rewards(R_j, c_j)
        
        self.throughput = R_j
        self.collision_info = c_j
        self.current_step += 1
        done = (self.current_step >= self.config.N)
        self.prev_actions = chosen_planes
        
        return self._get_states(), rewards, done

    def _assign_preambles(self, chosen_planes): #给每个UT分配前导码
        preambles = np.zeros(self.config.J, dtype=int)
        for j in range(self.config.J):
            if chosen_planes[j] > 0:
                preambles[j] = np.random.randint(1, self.config.P + 1)
        return preambles

    def _build_plane_preamble_map(self, chosen_planes, preambles): 
        #记录某（平面，前导码）被哪些UT选择了，用于冲突判断
        plane_preamble_map = {}
        for j in range(self.config.J):
            if chosen_planes[j] > 0:
                key = (chosen_planes[j], preambles[j])
                if key not in plane_preamble_map:
                    plane_preamble_map[key] = []
                plane_preamble_map[key].append(j)
        return plane_preamble_map

    def _compute_throughput_and_collisions(self, plane_preamble_map): #计算吞吐量和冲突率
        R_j = np.zeros(self.config.J)
        c_j = np.zeros(self.config.J)
        
        for _, uts in plane_preamble_map.items():
            if len(uts) > 1: #两个以上的UT选择了相同的（平面，前导码），发生冲突
                for u in uts:
                    c_j[u] = 1
            else:
                u = uts[0]
                R_j[u] = 1.0 #没冲突，则UT的吞吐量设为1（简化版）
        
        return R_j, c_j

    def _compute_rewards(self, R_j, c_j): #计算奖励
        return (R_j - self.config.rho * c_j - self.config.mu) / self.config.sigma

    def _get_states(self): #获取状态
        states = []
        for j in range(self.config.J):
            state = [
                self.current_step / self.config.N,
                self.prev_actions[j] / float(self.config.K),
                self.throughput[j],
                self.collision_info[j],
                self.sat_features[0],
                self.sat_features[1]
            ]
            states.append(state)
        return np.array(states)
