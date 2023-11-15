import nn

class PerceptronModel(object):
    def __init__(self, dimensions):
        """
        Initialize a new Perceptron instance.

        A perceptron classifies data points as either belonging to a particular
        class (+1) or not (-1). `dimensions` is the dimensionality of the data.
        For example, dimensions=2 would mean that the perceptron must classify
        2D points.
        """
        self.arr1 = nn.Parameter(1, dimensions)

    def get_weights(self):
        """
        Return a Parameter instance with the current weights of the perceptron.
        """
        return self.arr1
    def run(self, x):
        """
        Calculates the score assigned by the perceptron to a data point x.

        Inputs:
            x: a node with shape (1 x dimensions)
        Returns: a node containing a single number (the score)
        """
        "*** YOUR CODE HERE ***"
        return nn.DotProduct(x,self.arr1)
    def get_prediction(self, x):
        """
        Calculates the predicted class for a single data point `x`.

        Returns: 1 or -1
        """
        "*** YOUR CODE HERE ***"
        return 1 if nn.as_scalar(self.run(x))>=0 else -1
    def train(self, dataset):
        """
        Train the perceptron until convergence.
        """
        "*** YOUR CODE HERE ***"
        shStop = False
        while not shStop:
            for x, y in dataset.iterate_once(batch_size=1):
                if not self.get_prediction(x) == nn.as_scalar(y):
                    self.arr1.update(x, nn.as_scalar(y))
                    break
            else:
                shStop = True
class RegressionModel(object):
    """
    A neural network model for approximating a function that maps from real
    numbers to real numbers. The network should be sufficiently large to be able
    to approximate sin(x) on the interval [-2pi, 2pi] to reasonable precision.
    """
    def __init__(self):
        # Initialize your model parameters here
        "*** YOUR CODE HERE ***"
        self.in_features = 10
        self.hidden_features = 100
        self.rate = 0.3
        self.layer_number = 3
        self.arr1 = []
        self.arr2 = []
        for i in range(self.layer_number):
            if i==0:
                self.arr1.append(nn.Parameter(1,self.in_features))
                self.arr2.append(nn.Parameter(1,self.in_features))
            elif i==self.layer_number-1:
                self.arr1.append(nn.Parameter(self.in_features,1))
                self.arr2.append(nn.Parameter(1,1))
            else:
                self.arr1.append(nn.Parameter(self.in_features,self.in_features))
                self.arr2.append(nn.Parameter(1,self.in_features))
    def run(self, x):
        """
        Runs the model for a batch of examples.

        Inputs:
            x: a node with shape (batch_size x 1)
        Returns:
            A node with shape (batch_size x 1) containing predicted y-values
        """
        "*** YOUR CODE HERE ***"
        input_data = x
        for i in range(self.layer_number):
            fx = nn.Linear(input_data, self.arr1[i])
            output_data = nn.AddBias(fx, self.arr2[i])
            if i==self.layer_number-1:
                predict_y = output_data
            else:
                input_data = nn.ReLU(output_data)
        return predict_y
    def get_loss(self, x, y):
        """
        Computes the loss for a batch of examples.

        Inputs:
            x: a node with shape (batch_size x 1)
            y: a node with shape (batch_size x 1), containing the true y-values
                to be used for training
        Returns: a loss node
        """
        "*** YOUR CODE HERE ***"
        predict_y = self.run(x)
        return nn.SquareLoss(predict_y, y)
    def train(self, dataset):
        """
        Trains the model.
        """
        "*** YOUR CODE HERE ***"
        as_scalar_number = float('inf')
        count = 0
        while as_scalar_number>=0.01:
            for (x,y) in dataset.iterate_once(self.hidden_features):
                loss = self.get_loss(x,y)
                as_scalar_number = nn.as_scalar(loss)
                gradients = nn.gradients(loss, self.arr1+self.arr2)
                for i in range(self.layer_number):
                    self.arr1[i].update(gradients[i],-self.rate)
                    self.arr2[i].update(gradients[len(self.arr1)+i],-self.rate)
                count += 1
class DigitClassificationModel(object):
    """
    A model for handwritten digit classification using the MNIST dataset.

    Each handwritten digit is a 28x28 pixel grayscale image, which is flattened
    into a 784-dimensional vector for the purposes of this model. Each entry in
    the vector is a floating point number between 0 and 1.

    The goal is to sort each digit into one of 10 classes (number 0 through 9).

    (See RegressionModel for more information about the APIs of different
    methods here. We recommend that you implement the RegressionModel before
    working on this part of the project.)
    """
    def __init__(self):
        # Initialize your model parameters here
        "*** YOUR CODE HERE ***"
        self.layers = [200,500,10]
        self.batch_size = 100
        self.rate = 0.3
        self.arr1 = []
        self.arr2 = []
        for i in range(len(self.layers)):
            if i==0:
                self.arr1.append(nn.Parameter(784,self.layers[i]))
                self.arr2.append(nn.Parameter(1,self.layers[i]))
            elif i==len(self.layers)-1:
                self.arr1.append(nn.Parameter(self.layers[i-1],10))
                self.arr2.append(nn.Parameter(1,10))
            else:
                self.arr1.append(nn.Parameter(self.layers[i-1],self.layers[i]))
                self.arr2.append(nn.Parameter(1,self.layers[i]))
    def run(self, x):
        """
        Runs the model for a batch of examples.

        Your model should predict a node with shape (batch_size x 10),
        containing scores. Higher scores correspond to greater probability of
        the image belonging to a particular class.

        Inputs:
            x: a node with shape (batch_size x 784)
        Output:
            A node with shape (batch_size x 10) containing predicted scores
                (also called logits)
        """
        "*** YOUR CODE HERE ***"
        input_data = x
        for i in range(len(self.layers)):
            fx = nn.Linear(input_data, self.arr1[i])
            output_data = nn.AddBias(fx, self.arr2[i])
            if i==len(self.layers)-1:
                predict_y = output_data
            else:
                input_data = nn.ReLU(output_data)
        return predict_y
    def get_loss(self, x, y):
        """
        Computes the loss for a batch of examples.

        The correct labels `y` are represented as a node with shape
        (batch_size x 10). Each row is a one-hot vector encoding the correct
        digit class (0-9).

        Inputs:
            x: a node with shape (batch_size x 784)
            y: a node with shape (batch_size x 10)
        Returns: a loss node
        """
        "*** YOUR CODE HERE ***"
        predict_y = self.run(x)
        return nn.SoftmaxLoss(predict_y, y)
    def train(self, dataset):
        """
        Trains the model.
        """
        "*** YOUR CODE HERE ***"
        acc= 0
        while acc<0.98:
            for (x,y) in dataset.iterate_once(self.batch_size):
                loss = self.get_loss(x,y)
                gradients = nn.gradients(loss, self.arr1+self.arr2)
                for i in range(len(self.layers)):
                    self.arr1[i].update(gradients[i],-self.rate)
                    self.arr2[i].update(gradients[len(self.arr1)+i],-self.rate)
            acc= dataset.get_validation_accuracy()
            
class LanguageIDModel(object):
    """
    A model for language identification at a single-word granularity.

    (See RegressionModel for more information about the APIs of different
    methods here. We recommend that you implement the RegressionModel before
    working on this part of the project.)
    """
    def __init__(self):
        # Our dataset contains words from five different languages, and the
        # combined alphabets of the five languages contain a total of 47 unique
        # characters.
        # You can refer to self.num_chars or len(self.languages) in your code
        self.num_chars = 47
        self.languages = ["English", "Spanish", "Finnish", "Dutch", "Polish"]

        # Initialize your model parameters here
        "*** YOUR CODE HERE ***"
        self.rate = 0.2
        self.batch_size = 10
        self.threshold = 0.85
        self.hidden_size = 800

        self.arr1 = nn.Parameter(self.num_chars, self.hidden_size)
        self.arr2 = nn.Parameter(1,self.hidden_size)
        self.arr1_hidden = nn.Parameter(self.hidden_size, self.hidden_size)
        self.arr2_hidden = nn.Parameter(1, self.hidden_size)
        self.arr1_output = nn.Parameter(self.hidden_size, len(self.languages))
        self.arr2_output = nn.Parameter(1, len(self.languages))
    def run(self, xs):
        """
        Runs the model for a batch of examples.

        Although words have different lengths, our data processing guarantees
        that within a single batch, all words will be of the same length (L).

        Here `xs` will be a list of length L. Each element of `xs` will be a
        node with shape (batch_size x self.num_chars), where every row in the
        array is a one-hot vector encoding of a character. For example, if we
        have a batch of 8 three-letter words where the last word is "cat", then
        xs[1] will be a node that contains a 1 at position (7, 0). Here the
        index 7 reflects the fact that "cat" is the last word in the batch, and
        the index 0 reflects the fact that the letter "a" is the inital (0th)
        letter of our combined alphabet for this task.

        Your model should use a Recurrent Neural Network to summarize the list
        `xs` into a single node of shape (batch_size x hidden_size), for your
        choice of hidden_size. It should then calculate a node of shape
        (batch_size x 5) containing scores, where higher scores correspond to
        greater probability of the word originating from a particular language.

        Inputs:
            xs: a list with L elements (one per character), where each element
                is a node with shape (batch_size x self.num_chars)
        Returns:
            A node with shape (batch_size x 5) containing predicted scores
                (also called logits)
        """
        "*** YOUR CODE HERE ***"
        hidden_state_cur = nn.Linear(xs[0], self.arr1)
        for x in xs[1:]:
            l1 = nn.AddBias(nn.Linear(x, self.arr1), self.arr2)
            l2 = nn.AddBias(nn.Linear(hidden_state_cur, self.arr1_hidden), self.arr2_hidden)
            hidden_state_cur = nn.ReLU(nn.Add(l1, l2))
        y_predictions = nn.AddBias(nn.Linear(hidden_state_cur, self.arr1_output), self.arr2_output)
        return y_predictions
    def get_loss(self, xs, y):
        """
        Computes the loss for a batch of examples.

        The correct labels `y` are represented as a node with shape
        (batch_size x 5). Each row is a one-hot vector encoding the correct
        language.

        Inputs:
            xs: a list with L elements (one per character), where each element
                is a node with shape (batch_size x self.num_chars)
            y: a node with shape (batch_size x 5)
        Returns: a loss node
        """
        "*** YOUR CODE HERE ***"
        predict_y = self.run(xs)
        return nn.SoftmaxLoss(predict_y,y)
    def train(self, dataset):
        """
        Trains the model.
        """
        "*** YOUR CODE HERE ***"
        acc= 0
        while acc< self.threshold:
            for x, y in dataset.iterate_once(self.batch_size):
                loss = self.get_loss(x,y)
                values = [self.arr1,self.arr2,self.arr1_hidden,self.arr2_hidden,self.arr1_output,self.arr2_output]
                gradients = nn.gradients(loss, values)
                for i in range(len(values)):
                    value = values[i]
                    value.update(gradients[i], -self.rate)
            acc= dataset.get_validation_accuracy()