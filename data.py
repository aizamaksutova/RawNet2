import gdown

# a checkpoint
id_model = "1QQ5QP8gZa0Jwaur_JbHmvVW5gsBn3qbw"
output_model = "model_best.pth"
gdown.download(id=id_model, output=output_model, quiet=False)
