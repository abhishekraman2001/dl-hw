import torch


class NearestNeighborClassifier:
    """
    A class to perform nearest neighbor classification.
    """

    def __init__(self, x: list[list[float]], y: list[float]):
        """
        Store the data and labels to be used for nearest neighbor classification.
        You do not have to modify this function, but you will need to implement the functions it calls.
        https://pytorch.org/docs/stable/generated/torch.pow.html
        Args:
            x: list of lists of floats, data
            y: list of floats, labels
        """
        # Create and store data/labels as Tensors
        self.data, self.label = self.make_data(x, y)
        # Compute mean and std for each feature
        self.data_mean, self.data_std = self.compute_data_statistics(self.data)
        # Store the normalized version of the original data
        self.data_normalized = self.input_normalization(self.data)


    @classmethod
    def make_data(self, x: list[list[float]], y: list[float]) -> tuple[torch.Tensor, torch.Tensor]:
        
        """
        Warmup: Convert the data into PyTorch tensors.
        Assumptions:
        - len(x) == len(y)

        Args:
            x: list of lists of floats, data
            y: list of floats, labels
            - [torch.as_tensor](https://pytorch.org/docs/stable/generated/torch.as_tensor.html)
        Lecture 1.6
        """
        data = torch.tensor(x, dtype=torch.float32)
        label = torch.tensor(y, dtype=torch.float32)
  
        return data, label
        raise NotImplementedError

    @classmethod
    def compute_data_statistics(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Compute the mean and standard deviation of the data.
        Each row denotes a single data point.

        Args:
            x: 2D tensor data shape = [N, D]

        Returns:
            tuple of mean and standard deviation of the data.
            Both should have a shape [1, D]
        """
    
        mean = x.mean(dim=0)
        std = x.std(dim=0)  
        return mean, std
        #raise NotImplementedError

    def input_normalization(self, x: torch.Tensor) -> torch.Tensor:
        """
            Normalize input data using the stored mean and standard deviation:
            (data - mean) / (std + epsilon)
            Args:
            x: 1D or 2D tensor shape = [D] or [N, D]
        """
        eps = 1e-8  # small constant to avoid division by zero
        return (x - self.data_mean) / (self.data_std + eps)

    def get_nearest_neighbor(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Find the input x's nearest neighbor and the corresponding label.
        Args:
            x: 1D tensor of shape [D]

        Returns:
        A tuple (data_point, label) where:
            data_point is the nearest neighbor data point [D]
            label is the nearest neighbor's label [1] or a scalar tensor
        """
        # Normalize the input x using the same statistics from training
        x = self.input_normalization(x)
        dist = torch.norm(self.data_normalized - x, dim=1)  # shape [N]
    
        # Find the index of the smallest distance
        #https://pytorch.org/docs/stable/generated/torch.argmin.html
        idx = torch.argmin(dist)
    
        # Return the original (unnormalized) nearest neighbor data and label
        return self.data[idx], self.label[idx]

    def get_k_nearest_neighbor(self, x: torch.Tensor, k: int) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Find the k-nearest neighbors of input x from the training data.
        Args:
            x: A 1D tensor of shape [D]
            k: Number of neighbors to retrieve

        Returns:
            A tuple (neighbors, labels) where:
            neighbors is a [k, D] tensor of the k-nearest neighbors
            labels is a [k] tensor of the corresponding labels
            https://pytorch.org/docs/stable/generated/torch.topk.html
            Lecture KNN
        """
        # Normalize the input x using the same training statistics
        x_norm = self.input_normalization(x)
        dist = torch.norm(self.data_normalized - x_norm, dim=1)  # shape [N]
        # topk(..., largest=False) returns the smallest distances
        _, idx = torch.topk(dist, k, largest=False)
        # Return the original (unnormalized) data points and labels for the k indices
        return self.data[idx], self.label[idx]

    def knn_regression(self, x: torch.Tensor, k: int) -> torch.Tensor:
        """
        Use the k-nearest neighbors of the input x to predict its regression label.
        The prediction will be the average value of the labels from the k neighbors.
        https://pytorch.org/docs/stable/generated/torch.topk.html
        Args:
            x: 1D tensor [D]
            k: int, number of neighbors

        Returns:
            A tensor of shape [1] representing the average label value of the k neighbors.
        """
        # Normalize the input using the same training statistics
        x_norm = self.input_normalization(x)

        # Compute the L2 distance between x and each normalized training sample
        dist = torch.norm(self.data_normalized - x_norm, dim=1)  # shape [N]

        # Get the indices of the k smallest distances
        _, idx = torch.topk(dist, k, largest=False)

        # Gather the labels of these k neighbors
        knn_labels = self.label[idx]  # shape [k]

        # Compute the average value of these labels
        avg_label = knn_labels.mean()

        # Return it as a [1]-shaped tensor
        return avg_label.unsqueeze(0)
        raise NotImplementedError
