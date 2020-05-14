import unittest
import torch
import torch.nn as nn
import hyperLSTM


class MyTestCase(unittest.TestCase):
    def test_lstm_single(self):
        torch.manual_seed(42)
        pytorch_lstm = nn.LSTM(5, 2)
        lstm = hyperLSTM.LSTM(5, 2, forget_bias=0)
        lstm.wx.weight.data = dict(pytorch_lstm.named_parameters())['weight_ih_l0']
        lstm.wh.weight.data = dict(pytorch_lstm.named_parameters())['weight_hh_l0']
        lstm.wh.bias.data = dict(pytorch_lstm.named_parameters())['bias_ih_l0'] + dict(pytorch_lstm.named_parameters())['bias_hh_l0']

        input_ = torch.normal(torch.ones(1, 1, 5))
        state = (torch.ones(2) * 2, torch.ones(2) * 3)
        state_torch = (torch.ones(1, 1, 2) * 2, torch.ones(1, 1, 2) * 3)

        lstm_out = lstm(input_, state)
        pytorch_lstm = pytorch_lstm(input_, state_torch)

        self.assertTrue(torch.allclose(lstm_out[1][0].data, pytorch_lstm[1][0].data), "Short term memory differs")
        self.assertTrue(torch.allclose(lstm_out[1][1].data, pytorch_lstm[1][1].data), "Long term memory differs")

    def test_lstm_long(self):
        torch.manual_seed(42)
        pytorch_lstm = nn.LSTM(5, 2)
        lstm = hyperLSTM.LSTM(5, 2, forget_bias=0)
        lstm.wx.weight.data = dict(pytorch_lstm.named_parameters())['weight_ih_l0']
        lstm.wh.weight.data = dict(pytorch_lstm.named_parameters())['weight_hh_l0']
        lstm.wh.bias.data = dict(pytorch_lstm.named_parameters())['bias_ih_l0'] + dict(pytorch_lstm.named_parameters())['bias_hh_l0']

        input_ = torch.normal(torch.ones(10, 1, 5))
        state = (torch.ones(2) * 2, torch.ones(2) * 3)
        state_torch = (torch.ones(1, 1, 2) * 2, torch.ones(1, 1, 2) * 3)

        lstm_out = lstm(input_, state)
        pytorch_lstm = pytorch_lstm(input_, state_torch)

        self.assertTrue(torch.allclose(lstm_out[1][0].data, pytorch_lstm[1][0].data), "Short term memory differs")
        self.assertTrue(torch.allclose(lstm_out[1][1].data, pytorch_lstm[1][1].data), "Long term memory differs")


if __name__ == '__main__':
    unittest.main()
