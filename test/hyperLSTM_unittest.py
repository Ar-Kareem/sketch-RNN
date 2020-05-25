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

    def test_lstm_multple_states(self):
        torch.manual_seed(42)
        pytorch_lstm = nn.LSTM(5, 2)
        lstm = hyperLSTM.LSTM(5, 2, forget_bias=0)
        lstm.wx.weight.data = dict(pytorch_lstm.named_parameters())['weight_ih_l0']
        lstm.wh.weight.data = dict(pytorch_lstm.named_parameters())['weight_hh_l0']
        lstm.wh.bias.data = dict(pytorch_lstm.named_parameters())['bias_ih_l0'] + dict(pytorch_lstm.named_parameters())['bias_hh_l0']

        input_ = torch.normal(torch.ones(3, 10, 5))
        state = (torch.ones(10, 2) * 2, torch.ones(10, 2) * 3)
        state[0][0,1] = 2
        state[0][1,1] = 3
        state[1][0,0] = 4
        state[1][1,0] = 5
        state_torch = (state[0].clone().unsqueeze(0), state[1].clone().unsqueeze(0))

        lstm_out = lstm(input_, state)
        pytorch_lstm_out = pytorch_lstm(input_, state_torch)

        self.assertTrue(torch.allclose(lstm_out[1][0].data, pytorch_lstm_out[1][0].data), "Short term memory differs")
        self.assertTrue(torch.allclose(lstm_out[1][1].data, pytorch_lstm_out[1][1].data), "Long term memory differs")

        state[1][1,0] = 5.1
        lstm_out = lstm(input_, state)
        pytorch_lstm_out = pytorch_lstm(input_, state_torch)
        self.assertFalse(torch.allclose(lstm_out[1][0].data, pytorch_lstm_out[1][0].data), "Short term memory should be different")
        self.assertFalse(torch.allclose(lstm_out[1][1].data, pytorch_lstm_out[1][1].data), "Long term memory should be different")

    def test_lstm_empty_state(self):
        torch.manual_seed(42)
        pytorch_lstm = nn.LSTM(5, 2)
        lstm = hyperLSTM.LSTM(5, 2, forget_bias=0)
        lstm.wx.weight.data = dict(pytorch_lstm.named_parameters())['weight_ih_l0']
        lstm.wh.weight.data = dict(pytorch_lstm.named_parameters())['weight_hh_l0']
        lstm.wh.bias.data = dict(pytorch_lstm.named_parameters())['bias_ih_l0'] + dict(pytorch_lstm.named_parameters())['bias_hh_l0']

        input_ = torch.normal(torch.ones(3, 10, 5))
        state_torch = (torch.zeros(1, 10, 2), torch.zeros(1, 10, 2))

        lstm_out = lstm(input_)
        pytorch_lstm_out = pytorch_lstm(input_, state_torch)

        self.assertTrue(torch.allclose(lstm_out[1][0].data, pytorch_lstm_out[1][0].data), "Short term memory differs")
        self.assertTrue(torch.allclose(lstm_out[1][1].data, pytorch_lstm_out[1][1].data), "Long term memory differs")

    def test_hyper_lstm_not_crashing(self):
        torch.manual_seed(42)
        lstm = hyperLSTM.HyperLSTM(5, 2, layer_norm=True, dropout=0.1)

        input_ = torch.normal(torch.ones(3, 10, 5))

        lstm_out = lstm(input_)

if __name__ == '__main__':
    unittest.main()
