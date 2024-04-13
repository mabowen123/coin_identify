class PredictBase:
    def predict(self, *args, **kwargs):
        raise NotImplementedError


class ProcessBase:
    def preprocess(self, *args, **kwargs):
        raise NotImplementedError

    def post_process(self, *args, **kwargs):
        raise NotImplementedError