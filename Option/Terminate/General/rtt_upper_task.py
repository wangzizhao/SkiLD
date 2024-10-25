from Option.Terminate import RewardTerminateTruncate


class RTTUpperTask(RewardTerminateTruncate):
    """
    passes back all the true values
    """
    def rew(self, batch):
        return batch.env_rew

    def term(self, batch):
        return batch.terminated

    def trunc(self, batch):
        return batch.truncated
