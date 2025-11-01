# detection_alignment.py

class DetectionAlignment:
    """
    A class to calculate the Detection Alignment (DA) metric, which measures the 
    alignment between predicted and ground truth vulnerable lines in a dataset. 
    By convention, a sample is considered vulnerable if y_pred = 1.
    """
    def __init__(self):
        """
        This method sets up two lists, `inter_all` and `union_all`, that will 
        store the intersection and union values for each sample to calculate 
        the final DA score.
        """
        self.inter_all = []
        self.union_all = []

    def _get_sample_da(self, lines:list, scores:list, ground_truth:set, y_pred:int):
        """
        Calculate the DA for a single sample.
        The method computes the intersection and union between the predicted 
        vulnerable lines (`lines`, along their scores) and the ground truth (`ground_truth`).
        The DA is calculated differently based on the predicted label `y_pred`.

        Args:
            lines (list): list of lines in the sample
            scores (list): scores associated with each line
            ground_truth (set): set of ground truth vulnerable lines
            y_pred (int): predicted label for the sample (1 for vulnerable, 0 for non-vulnerable)

        Returns:
            tuple: A tuple containing the sum of intersection and the sum of union for the sample.
        """        ''''''
        intersection = []
        union = []

        # If the model predicts the sample as vulnerable, we compute the intersection normally
        if y_pred == 1:
            for line, line_score in zip(lines, scores):
                intersection.append(min(1 if line in ground_truth else 0, line_score))
        else:  # If the model predicts the sample as non-vulnerable, intersection is 0
            scores = [0]*len(scores)

        for line, line_score in zip(lines, scores):
            union.append(max(1 if line in ground_truth else 0, line_score))
        
        return sum(intersection), sum(union)

    def update(self, lines:list, scores:list, ground_truth:set, y_pred:int):
        """
        Update the Detection Alignment metric by appending the intersection and union 
        values for the current sample to the `inter_all` and `union_all` lists.

        This method is called for each sample to accumulate intersection and union values 
        across all samples.

        Args:
            lines (list): list of lines in the sample
            scores (list): scores associated with each line
            ground_truth (set): set of ground truth vulnerable lines
            y_pred (int): predicted label for the sample (1 for vulnerable, 0 for non-vulnerable)
        """
        intersection, union = self._get_sample_da(lines=lines,
                                                  scores=scores,
                                                  ground_truth=ground_truth,
                                                  y_pred=y_pred)
        self.inter_all.append(intersection)
        self.union_all.append(union)
    
    def get_da(self):
        """
        Calculate the final Detection Alignment (DA) score. The DA score is
        the ratio of the total intersection over the total union across all samples.

        Returns:
            float: The calculated DA score as a float.
        """
        da = sum(self.inter_all) / sum(self.union_all) if self.union_all else 0
        return da