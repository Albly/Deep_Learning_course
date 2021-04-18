# Don't erase the template code, except "Your code here" comments.

import torch
import math                      # Pi

""" Task 1 """

def get_rho():
    theta = torch.linspace(-math.pi,math.pi, 1000)
    assert theta.shape == (1000,)

    # (2) Your code here; rho = ...
    rho = (1+0.9*torch.cos(8*theta))*(1+0.1*torch.cos(24*theta))*(0.9+0.05*torch.cos(200*theta)) *(1+torch.sin(theta)) 
    assert torch.is_same_size(rho, theta)
    
    # (3) Your code here; x = ...
    x = rho * torch.cos(theta)
    # (3) Your code here; y = ...
    y= rho * torch.sin(theta)
    return x, y

""" Task 2 """

def game_of_life_update_torch(alive_map):
    """
    PyTorch version of `game_of_life_update_reference()`.
    
    alive_map:
        `torch.tensor`, ndim == 2, dtype == `torch.int64`
        The game map containing 0s (dead) an 1s (alive).
    """
    #kernel that counts number of alived neighbour cells 
    kernel = torch.tensor([[1,1,1],
                           [1,0,1],
                           [1,1,1]], dtype = torch.int64).reshape(1,1,3,3)
    
    #increase num of dims for conv operation. Needed shape is [N,ch,H,W]
    # where N - batches, ch -channels, h -height , w - width
    alive_map = alive_map.unsqueeze(dim=0).unsqueeze(dim=0)

    # do conv operation
    n_alived_cells = torch.conv2d(input = alive_map,
                               weight = kernel,
                               stride = 1,
                               padding=1)
    
    # if a cell was dead and it has 3 alived neighbours -> make it alive 
    borned_cells = (n_alived_cells == 3) & (alive_map == 0)
    # if a cell was alived and it has 2 or 3 alived neighbours -> stil alived cell
    survived_cells = ((n_alived_cells == 2) | (n_alived_cells == 3)) & (alive_map == 1)
    # Apply both conditions
    new_alive_map = survived_cells | borned_cells
    
    alive_map.copy_(new_alive_map.squeeze(dim = 0).squeeze(dim = 0)) 
    
    

""" Task 3 """

# This is a reference layout for encapsulating your neural network. You can add arguments and
# methods if you need to. For example, you may want to add a method `do_gradient_step()` that
# executes one step of an optimization algorithm (SGD / Adadelta / Adam / ...); or you can
# add an extra argument for which you'll find a good value during experiments and set it as
# default to preserve interface (e.g. `def __init__(self, num_hidden_neurons=100):`).
from IPython import display
import matplotlib.pyplot as plt

class NeuralNet:
    def __init__(self, input_size = 28*28, output_size = 10, hidden_1 = 150):
        # Your code here
        # 1st linear layer
        self.W1 = torch.randn(input_size, hidden_1, requires_grad = True)
        self.b1 = torch.randn(hidden_1, requires_grad = True)
        # 2nd linear layer
        self.W2 = torch.randn(hidden_1, output_size, requires_grad = True)
        self.b2 = torch.randn(output_size, requires_grad = True)

        #saved params for Adam optivizer
        self.state = {}

    def zero_grad(self):
        # make all param's grad equal to 0
        for param in [self.W1, self.b1, self.W2, self.b2]:
            param.grad.zero_()

    def adam_step(self, lr = torch.tensor(1e-3), beta1 = torch.tensor(0.9), beta2 = torch.tensor(0.999), eps = 1e-8):
        #Adam optimizer
        with torch.no_grad():
            self.state.setdefault('m', {})
            self.state.setdefault('v', {})
            self.state.setdefault('t', 0)
            self.state['t'] += 1

            var_index = 0
            lr_t = lr * torch.sqrt(1 - beta2**self.state['t']) / (1 - beta1**self.state['t'])
            for param in [self.W1, self.b1, self.W2, self.b2]:
                mu = self.state['m'].setdefault(var_index, torch.zeros_like(param.grad))
                v = self.state['v'].setdefault(var_index, torch.zeros_like(param.grad))

                torch.add(beta1 * mu , (1-beta1) * param.grad, out = mu) 
                torch.add(beta2 * v , (1-beta2) *  param.grad * param.grad, out= v) 
                param -= lr_t * mu/(torch.sqrt(v) + eps) 

                assert mu is self.state['m'].get(var_index)
                assert v is self.state['v'].get(var_index)
                var_index +=1
            

    def predict(self, images):
        """
        images:
            `torch.tensor`, shape == `batch_size x height x width`, dtype == `torch.float32`
            A minibatch of images -- the input to the neural net.
        
        return:
        prediction:
            `torch.tensor`, shape == `batch_size x 10`, dtype == `torch.float32`
            The scores of each input image to belong to each of the dataset classes.
            Namely, `prediction[i, j]` is the score of `i`-th minibatch sample to
            belong to `j`-th class.
            These scores can be 0..1 probabilities, but for better numerical stability
            they can also be raw class scores after the last (usually linear) layer,
            i.e. BEFORE softmax.
        """
        # Your code here
        # input shapes
        batch_size, height, width = images.shape
        # reshaping
        images = images.view(batch_size, height*width)
        # first linear layer
        self.out = torch.matmul(images, self.W1) + self.b1
        # activation
        self.out = torch.relu(self.out)
        # second linear layers
        self.out = torch.matmul(self.out, self.W2) + self.b2
        # logsoftmax
        self.out = torch.log_softmax(self.out, dim = 1)
        return self.out

    # Your code here

def accuracy(model, images, labels):
    """
    Use `NeuralNet.predict` here.
    
    model:
        `NeuralNet`
    images:
        `torch.tensor`, shape == `N x height x width`, dtype == `torch.float32`
    labels:
        `torch.tensor`, shape == `N`, dtype == `torch.int64`
        Class indices for each sample in `images`.
    
    return:
    value:
        `float`
        The fraction of samples from `images` correctly classified by `model`.
        `0 <= value <= 1`.
    """

    preds = model.predict(images)
    idxs = torch.argmax(preds, dim = 1)
    return torch.mean((idxs == labels).type(torch.float32))

    # Your code here

def train_on_notmnist(model, X_train, y_train, X_val, y_val):
    """
    Update `model`'s weights so that its accuracy on `X_val` is >=82%.
    `X_val`, `y_val` are provided for convenience and aren't required to be used.
    
    model:
        `NeuralNet`
    X_train:
        `torch.tensor`, shape == `N x height x width`, dtype == `torch.float32`
    y_train:
        `torch.tensor`, shape == `N`, dtype == `torch.int64`
        Class indices for each sample in `X_train`.
    X_val, y_val:
        Same as above, possibly with a different length.
    """
    # batch generator
    def get_batches(dataset, batch_size):
        X, Y = dataset
        n_samples = X.shape[0]
            
        # Shuffle at the start of epoch
        indices = torch.randperm(n_samples)
        
        for start in range(0, n_samples, batch_size):
            end = min(start + batch_size, n_samples)
            
            batch_idx = indices[start:end]
        
            yield X[batch_idx], Y[batch_idx]

    #One hot encoder transformation
    def one_hot_transform(y):
        y_hot = torch.zeros(len(y), len(y.unique()))
        y_hot.scatter_(1, y.unsqueeze(-1), 1)
        return y_hot

    # loss function
    def neg_log_likelihood(y_pred, y_true):
        return - torch.sum(torch.sum(y_pred * y_true, dim = 1), dim = 0) / y_pred.shape[0] 
    
    n_epochs = 100
    loss_history = []
    score_history = []
    batch_size = 300

    y_train_hot = one_hot_transform(y_train)
    y_val_hot = one_hot_transform(y_val)

    # for each epoch
    for epoch in range(n_epochs):
        # get data batches
        for x_batch, y_batch in get_batches((X_train, y_train_hot), batch_size):
            # make predicts
            preds = model.predict(x_batch)
            # calculate loss
            loss = neg_log_likelihood(preds, y_batch)
            # calculate gradients
            loss.backward()
            # change weights
            model.adam_step()
            # clear grads
            model.zero_grad

            # add loss to history
            loss_history.append(loss)
    
        # for val set
        display.clear_output(wait=True)
        # calculate accuracy
        score = accuracy(model, X_val, y_val)
        print(f'Accuracy:{score}')
        
        #plot picture with loss
        plt.figure(figsize = (8,6))
        plt.title("Negative log likelihood", fontsize = 16)
        plt.xlabel("Iteration",fontsize = 14)
        plt.ylabel("Loss",fontsize = 14)
        #plt.yscale('log')
        plt.plot(loss_history, label='Train')
        plt.grid(alpha = 0.5)
        plt.legend()
        plt.show()

        if score >= 0.82:
            print('Required score is achieved')
            print('Stopping...')
            break 