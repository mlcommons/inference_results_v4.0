CORES_PER_SOCKET = 60
NUM_SOCKETS = 2

class InputItem:
    def __init__(
        self,
        id_list,
        idx_list,
        samples=None,
        data=None,
        labels=None,
        input_tokens=None,
        input_tokens_2=None,
        latents=None,
        receipt_time=0,
    ):
        self.id_list = id_list
        self.idx_list = idx_list
        self.samples = samples
        self.data = data
        self.labels = labels
        self.input_tokens = input_tokens
        self.input_tokens_2 = input_tokens_2
        self.latents = latents
        self.receipt_time = receipt_time
        

class OutputItem:
    def __init__(self, id, result):
        self.id = id
        self.result = result
