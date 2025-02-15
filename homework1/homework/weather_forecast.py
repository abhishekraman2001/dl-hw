from typing import Tuple
import torch

class WeatherForecast:
    def __init__(self, data_raw: list[list[float]]):
        """
        You are given a list of 10 weather measurements per day.
        Save the data as a PyTorch (num_days, 10) tensor,
        where the first dimension represents the day,
        and the second dimension represents the measurements.
        https://pytorch.org/docs/stable/generated/torch.arange.html
        https://pytorch.org/docs/stable/generated/torch.Tensor.float.html
        """
        self.data = torch.as_tensor(data_raw, dtype=torch.float32).view(-1, 10)


    def find_min_and_max_per_day(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Find the max and min temperatures per day

        Returns:
            min_per_day: tensor of size (num_days,)
            max_per_day: tensor of size (num_days,)
        """
        # Minimum along each row (dim=1)
        min_pday = self.data.min(dim=1).values
        # Maximum along each row (dim=1)
        max_pday = self.data.max(dim=1).values
        return min_pday, max_pday
        raise NotImplementedError

    def find_the_largest_drop(self) -> torch.Tensor:
        """
        Find the largest change in day-over-day average temperature.
        This should be a negative number.

        We compute average temperatures for each day, then calculate
        the difference (day[i+1] - day[i]). The largest drop is the most
        negative difference.

        Returns:
            tensor of a single value, the difference in temperature
        """
        # Compute average temperature per day => shape: (num_days,)
        day_ave = self.data.mean(dim=1)
        # Day-to-day differences: day_averages[i+1] - day_averages[i]
        dtod_diffs = day_ave[1:] - day_ave[:-1]
        # The largest drop is the minimum of these differences (most negative)
        return dtod_diffs.min()
        raise NotImplementedError

    def find_the_most_extreme_day(self) -> torch.Tensor:
        """
        For each day, find the measurement that differs the most
        from the day's average temperature.

        Returns:
            tensor with size (num_days,)
            (the value of the measurement that is farthest from the average)
        """
        # Averages per row => shape: (num_days, 1)
        day_ave = self.data.mean(dim=1, keepdim=True)
        # Absolute differences from the average => shape: (num_days, 10)
        diff = (self.data - day_ave).abs()
        # Index of the max difference per row => shape: (num_days,)
        max_diff_indices = diff.argmax(dim=1)
        # Gather those most-extreme measurements from the original data
        most_extreme = self.data.gather(1, max_diff_indices.unsqueeze(1)).squeeze(1)
        return most_extreme
        raise NotImplementedError

    def max_last_k_days(self, k: int) -> torch.Tensor:
        """
        Find the maximum temperature over the last k days.

        Returns:
            tensor of size (k,)
            (each entry is the max temperature on that day)
        """
        # Last k rows => shape: (k, 10)
        last_k = self.data[-k:]
        # Max along each row => shape: (k,)
        return last_k.max(dim=1).values
        raise NotImplementedError

    def predict_temperature(self, k: int) -> torch.Tensor:
        """
        From the dataset, predict the temperature of the next day.
        The prediction will be the average of all temperatures over the past k days.

        Args:
            k: int, number of days to consider

        Returns:
            tensor of a single value, the predicted temperature
        """
        # Last k days => shape: (k, 10)
        last_k_d = self.data[-k:]
        # Average of all measurements in these k days => shape: single value
        predicted_temp = last_k_d.mean()
        return predicted_temp

        raise NotImplementedError

    def what_day_is_this_from(self, t: torch.FloatTensor) -> torch.LongTensor:
        """
        You find a phone with one full day's temperature measurements (10 values).
        Determine which day in self.data is the closest match by
        sum of absolute differences.

        Args:
            t: tensor of size (10,), temperature measurements

        Returns:
            tensor of a single value, the index of the closest data element (0-based)
        """
        # Expand t if needed => shape: (1, 10)
        t = t.view(1, 10)
        # Compute sum of absolute differences between t and every row in self.data
        # shape of diffs => (num_days,)
        diffs = (self.data - t).abs().sum(dim=1)
        # argmin gives the index of the row with the smallest difference
        closest_day = diffs.argmin()
        return closest_day
        raise NotImplementedError
