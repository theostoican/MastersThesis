import numpy as np

from utils.models import DummyModel, TeacherModel
from utils.training import train_second_order

class SaddleFinder:
    def __init__(self, incoming_weights_x, incoming_weights_y,
                 outgoing_weights):
        self.network_size = len(incoming_weights_x)
        
        self.incoming_weights_x = incoming_weights_x
        self.incoming_weights_y = incoming_weights_y
        self.outgoing_weights = outgoing_weights

    def get_merged_neuron_point():
        min_dist = np.inf
        idx_neuron1 = None
        idx_neuron2 = None

        for i in range(self.network_size):
          current_neuron = np.array([self.incoming_weights_x[i],
                                     self.incoming_weights_y[i]])
          for j in range(i + 1, self.network_size):
            potential_closest_neuron = np.array([self.incoming_weights_x[j],
                                                 self.incoming_weights_y[j]])
            if min_dist > np.linalg.norm(current_neuron - potential_closest_neuron):
              min_dist = np.linalg.norm(current_neuron - potential_closest_neuron)
              idx_neuron1 = i
              idx_neuron2 = j

        new_incoming_weights_x = [(incoming_weights_x[idx_neuron1] + \
                                   incoming_weights_x[idx_neuron2]) / 2.0]
        new_incoming_weights_y = [(incoming_weights_y[idx_neuron1] + \
                                   incoming_weights_y[idx_neuron2]) / 2.0]
        new_outgoing_weights = [(outgoing_weights[idx_neuron1] + \
                                 outgoing_weights[idx_neuron2])]

        for i in range(self.network_size):
          if i == idx_neuron1 or i == idx_neuron2:
            continue
          new_incoming_weights_x.append(incoming_weights_x[i])
          new_incoming_weights_y.append(incoming_weights_y[i])
          new_outgoing_weights.append(outgoing_weights[i])
        
        return new_incoming_weights_x, new_incoming_weights_y, new_outgoing_weights


    def find_saddle():
        pass
    
class ReducedNetSaddleFinder(SaddleFinder):
    def find_saddle():        
        merged_incoming_weights_x, \
        merged_incoming_weights_y, \
        merged_outgoing_weights = get_merged_neuron_point()
        
        w_in = torch.DoubleTensor([[new_incoming_weights_x[0], new_incoming_weights_y[0]],
                           [new_incoming_weights_x[1], new_incoming_weights_y[1]],
                           [new_incoming_weights_x[2], new_incoming_weights_y[2]],
                           [new_incoming_weights_x[3], new_incoming_weights_y[3]]])
        w_out = torch.DoubleTensor([new_outgoing_weights])
        dummy_model = DummyNetwork(D_in, H_teacher, D_out, w_in, w_out)
        
        final_weights = train_second_order(dummy_model)


        

class MergerReducedNetSaddleFinder(SaddleFinder):
    pass
