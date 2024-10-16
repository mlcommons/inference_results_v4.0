import time

class InputItem:
    def __init__(self, id_list, index_list, input_token_ids=None, input_seq_lens=None, data=None, label=None, receipt_time=0):
        self.query_id_list = id_list
        self.sample_index_list = index_list
        self.data = data
        self.label = label
        self.receipt_time = receipt_time
        self.input_seq_lens = input_seq_lens

    def set_receipt_time(self, receipt_time):
        self.receipt_time = receipt_time


class OutputItem:
    def __init__(self, query_id_list, result, input_lengths=[], array_type_code='B'):
        self.query_id_list = query_id_list
        self.result = result
        self.array_type_code = array_type_code
        self.receipt_time = None
        self.outqueue_time = None
        self.input_lengths = input_lengths

    def set_receipt_time(self, receipt_time):
        self.receipt_time = receipt_time

    def set_outqueued_time(self, outqueue_time):
        self.outqueue_time = outqueue_time
