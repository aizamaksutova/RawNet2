import numpy as np
import torch
import hw_tts.waveglow.inference
import hw_tts.text as text


def synth(model, device, waveglow_model, text, alpha=1.0, p_alpha=1.0, e_alpha=1.0, path=None):
    text = np.array(text)
    text = np.stack([text])
    src_pos = np.array([i + 1 for i in range(text.shape[1])])
    src_pos = np.stack([src_pos])
    sequence = torch.from_numpy(text).long().to(device)
    src_pos = torch.from_numpy(src_pos).long().to(device)

    with torch.no_grad():
        mel = model.forward(sequence, src_pos, alpha=alpha, p_alpha=p_alpha, e_alpha=e_alpha)

    mel_cpu, mel_cuda = mel[0].cpu().transpose(0, 1), mel.contiguous().transpose(1, 2)
    if path is None:
        return hw_tts.waveglow.inference.get_wav(mel_cuda, waveglow_model)
    else:
        hw_tts.waveglow.inference.inference(mel_cuda, waveglow_model, path)

def get_text_data(user_tests=None):
    tests = [ 
        "A defibrillator is a device that gives a high energy electric shock to the heart of someone who is in cardiac arrest",
        "Massachusetts Institute of Technology may be best known for its math, science and engineering education",
        "Wasserstein distance or Kantorovich Rubinstein metric is a distance function defined between probability distributions on a given metric space"
    ]
    if user_tests is not None:
        tests = user_tests
    data_list = list(text.text_to_sequence(test, ['english_cleaners']) for test in tests)

    return data_list