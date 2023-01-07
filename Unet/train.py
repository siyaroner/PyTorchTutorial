{
  "nbformat": 4,
  "nbformat_minor": 0,
  "metadata": {
    "colab": {
      "provenance": []
    },
    "kernelspec": {
      "name": "python3",
      "display_name": "Python 3"
    },
    "language_info": {
      "name": "python"
    }
  },
  "cells": [
    {
      "cell_type": "code",
      "source": [
        "!pip install import-ipynb\n",
        "import import_ipynb\n"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "BMIWGsKuXcJ5",
        "outputId": "8ca00974-7500-402f-8870-25a57f1215c9"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Looking in indexes: https://pypi.org/simple, https://us-python.pkg.dev/colab-wheels/public/simple/\n",
            "Collecting import-ipynb\n",
            "  Downloading import_ipynb-0.1.4-py3-none-any.whl (4.1 kB)\n",
            "Requirement already satisfied: IPython in /usr/local/lib/python3.8/dist-packages (from import-ipynb) (7.9.0)\n",
            "Requirement already satisfied: nbformat in /usr/local/lib/python3.8/dist-packages (from import-ipynb) (5.7.0)\n",
            "Requirement already satisfied: decorator in /usr/local/lib/python3.8/dist-packages (from IPython->import-ipynb) (4.4.2)\n",
            "Requirement already satisfied: prompt-toolkit<2.1.0,>=2.0.0 in /usr/local/lib/python3.8/dist-packages (from IPython->import-ipynb) (2.0.10)\n",
            "Requirement already satisfied: backcall in /usr/local/lib/python3.8/dist-packages (from IPython->import-ipynb) (0.2.0)\n",
            "Requirement already satisfied: pexpect in /usr/local/lib/python3.8/dist-packages (from IPython->import-ipynb) (4.8.0)\n",
            "Requirement already satisfied: setuptools>=18.5 in /usr/local/lib/python3.8/dist-packages (from IPython->import-ipynb) (57.4.0)\n",
            "Collecting jedi>=0.10\n",
            "  Downloading jedi-0.18.2-py2.py3-none-any.whl (1.6 MB)\n",
            "\u001b[K     |████████████████████████████████| 1.6 MB 4.9 MB/s \n",
            "\u001b[?25hRequirement already satisfied: pickleshare in /usr/local/lib/python3.8/dist-packages (from IPython->import-ipynb) (0.7.5)\n",
            "Requirement already satisfied: traitlets>=4.2 in /usr/local/lib/python3.8/dist-packages (from IPython->import-ipynb) (5.7.1)\n",
            "Requirement already satisfied: pygments in /usr/local/lib/python3.8/dist-packages (from IPython->import-ipynb) (2.6.1)\n",
            "Requirement already satisfied: parso<0.9.0,>=0.8.0 in /usr/local/lib/python3.8/dist-packages (from jedi>=0.10->IPython->import-ipynb) (0.8.3)\n",
            "Requirement already satisfied: wcwidth in /usr/local/lib/python3.8/dist-packages (from prompt-toolkit<2.1.0,>=2.0.0->IPython->import-ipynb) (0.2.5)\n",
            "Requirement already satisfied: six>=1.9.0 in /usr/local/lib/python3.8/dist-packages (from prompt-toolkit<2.1.0,>=2.0.0->IPython->import-ipynb) (1.15.0)\n",
            "Requirement already satisfied: jsonschema>=2.6 in /usr/local/lib/python3.8/dist-packages (from nbformat->import-ipynb) (4.3.3)\n",
            "Requirement already satisfied: jupyter-core in /usr/local/lib/python3.8/dist-packages (from nbformat->import-ipynb) (5.1.0)\n",
            "Requirement already satisfied: fastjsonschema in /usr/local/lib/python3.8/dist-packages (from nbformat->import-ipynb) (2.16.2)\n",
            "Requirement already satisfied: pyrsistent!=0.17.0,!=0.17.1,!=0.17.2,>=0.14.0 in /usr/local/lib/python3.8/dist-packages (from jsonschema>=2.6->nbformat->import-ipynb) (0.19.2)\n",
            "Requirement already satisfied: attrs>=17.4.0 in /usr/local/lib/python3.8/dist-packages (from jsonschema>=2.6->nbformat->import-ipynb) (22.1.0)\n",
            "Requirement already satisfied: importlib-resources>=1.4.0 in /usr/local/lib/python3.8/dist-packages (from jsonschema>=2.6->nbformat->import-ipynb) (5.10.1)\n",
            "Requirement already satisfied: zipp>=3.1.0 in /usr/local/lib/python3.8/dist-packages (from importlib-resources>=1.4.0->jsonschema>=2.6->nbformat->import-ipynb) (3.11.0)\n",
            "Requirement already satisfied: platformdirs>=2.5 in /usr/local/lib/python3.8/dist-packages (from jupyter-core->nbformat->import-ipynb) (2.6.0)\n",
            "Requirement already satisfied: ptyprocess>=0.5 in /usr/local/lib/python3.8/dist-packages (from pexpect->IPython->import-ipynb) (0.7.0)\n",
            "Installing collected packages: jedi, import-ipynb\n",
            "Successfully installed import-ipynb-0.1.4 jedi-0.18.2\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 373
        },
        "id": "NGiDSx1FGhil",
        "outputId": "a3096bee-7258-4aa2-d469-cc9f5b5a7e34"
      },
      "outputs": [
        {
          "output_type": "error",
          "ename": "ModuleNotFoundError",
          "evalue": "ignored",
          "traceback": [
            "\u001b[0;31m---------------------------------------------------------------------------\u001b[0m",
            "\u001b[0;31mModuleNotFoundError\u001b[0m                       Traceback (most recent call last)",
            "\u001b[0;32m<ipython-input-5-a56b60287500>\u001b[0m in \u001b[0;36m<module>\u001b[0;34m\u001b[0m\n\u001b[1;32m      5\u001b[0m \u001b[0;32mimport\u001b[0m \u001b[0mtorch\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mnn\u001b[0m \u001b[0;32mas\u001b[0m \u001b[0mnn\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m      6\u001b[0m \u001b[0;32mimport\u001b[0m \u001b[0mtorch\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0moptim\u001b[0m \u001b[0;32mas\u001b[0m \u001b[0moptim\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0;32m----> 7\u001b[0;31m \u001b[0;32mfrom\u001b[0m \u001b[0mmodel\u001b[0m \u001b[0;32mimport\u001b[0m \u001b[0mUnet\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0m\u001b[1;32m      8\u001b[0m \u001b[0;31m# from utils import (\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m      9\u001b[0m \u001b[0;31m#     load_checkpoint,\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n",
            "\u001b[0;31mModuleNotFoundError\u001b[0m: No module named 'model'",
            "",
            "\u001b[0;31m---------------------------------------------------------------------------\u001b[0;32m\nNOTE: If your import is failing due to a missing package, you can\nmanually install dependencies using either !pip or !apt.\n\nTo view examples of installing some common dependencies, click the\n\"Open Examples\" button below.\n\u001b[0;31m---------------------------------------------------------------------------\u001b[0m\n"
          ],
          "errorDetails": {
            "actions": [
              {
                "action": "open_url",
                "actionText": "Open Examples",
                "url": "/notebooks/snippets/importing_libraries.ipynb"
              }
            ]
          }
        }
      ],
      "source": [
        "import torch\n",
        "import albumentations as A\n",
        "from albumentations.pytorch import ToTensorV2\n",
        "from tqdm import tqdm\n",
        "import torch.nn as nn\n",
        "import torch.optim as optim\n",
        "from model import Unet\n",
        "from utils import (\n",
        "    load_checkpoint,\n",
        "    save_checkpoint,\n",
        "    get_loaders,\n",
        "    check_accuracy,\n",
        "    save_predictions_as_imgs,\n",
        ")"
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "# Hyeperparameters\n",
        "LEARNING_RATE=1e-4\n",
        "DEVICE= \"cuda\" if torch.cuda.is_available() else \"cpu\"\n",
        "BATCH_SIZE=16\n",
        "NUM_EPOCHS=3\n",
        "NUM_WORKERS=2\n",
        "IMAGE_HEIGHT=160 #1289 ORIGINALLY\n",
        "IMAGE_WIDTH=240 #1918 ORIGINALLY\n",
        "PIN_MEMORY=True\n",
        "LOAD_MODEL=False\n",
        "TRAIN_IMG_DIR=\"data/train_images/\"\n",
        "TRAIN_MASK_DIR=\"data/train_masks/\"\n",
        "VAL_IMG_DIR=\"data/val_images/\"\n",
        "VAL_MASK_DIR=\"data/val_masks/\""
      ],
      "metadata": {
        "id": "MPXlURMIHzBJ"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "from torch.cuda import device_of\n",
        "from albumentations.augmentations.transforms import VerticalFlip\n",
        "def train_fn(loader,model,optimizer,loss_fn,scaler):\n",
        "  loop=tqdm(loader)\n",
        "  for batch_idx,(data,targets) in enumerate(loop):\n",
        "    data=data.to(device=DEVICE)\n",
        "    targets=targets.flooat().unsqueenze(1).to(device=DEVICE)\n",
        "\n",
        "    #FORWARD\n",
        "    with torch.cuda.amp.autocast():\n",
        "      predictions=model(data)\n",
        "      loss=loss_fn(predictions,targets)\n",
        "\n",
        "    #backward\n",
        "    optimizer.zero_grad()\n",
        "    scaler.scale(loss).backward()\n",
        "    scaler.step(optimizer)\n",
        "    scaler.update()\n",
        "\n",
        "    #update tqdm loop\n",
        "    loop.set_postfix(loss=loss.item())\n",
        "def main():\n",
        "  train_transform=A.Compose(\n",
        "      [\n",
        "          A.Resize(height=IMAGE_HEIGHT,width=IMAGE_WIDTH)\n",
        "          A.Rotate(limit=35,p=1.0)\n",
        "          A.HorizontalFlip(p=0.5)\n",
        "          A.VerticalFlip(p=0.1)\n",
        "          A.Normalize(\n",
        "              mean=[0.0,0.0,0.0],\n",
        "              std=[1.0,1.0,1.0],\n",
        "              max_pixel_value=255.0\n",
        "          ),\n",
        "       ToTensorV2(),\n",
        "      ],\n",
        "  )\n",
        "  val_transfroms=A.Compose(\n",
        "      [\n",
        "          A.Resize(height=IMAGE_HEIGHT,width=IMAGE_WIDTH)\n",
        "          A.Normalize(\n",
        "              mean=[0.0,0.0,0.0],\n",
        "              std=[1.0,1.0,1.0],\n",
        "              max_pixel_value=255.0\n",
        "          ),\n",
        "       ToTensorV2(),\n",
        "      ],\n",
        "  )\n",
        "\n",
        "  model=Umet(in_channels=1).to(DEVICE)\n",
        "  loss_fn=nn.BCEWithLogitsLoss()\n",
        "  optimizer=optim.Adam(model.parameters(),lr=LEARNING_RATE)\n",
        "\n",
        "  train_loader,val_loader=get_loaders(\n",
        "      TRAIN_IMG_DIR,\n",
        "      TRAIN_MASK_DIR,\n",
        "      VAL_IMG_DIR,\n",
        "      VAL_MASK_DIR\n",
        "      BATCH_SIZE,\n",
        "      train_transform,\n",
        "      val_transform,\n",
        "      NUM_WORKERS,\n",
        "      PIN_MEMORY\n",
        "  )\n",
        "\n",
        "  if LOAD_MODEL:\n",
        "    load_checkpoint(torch.load(\"my_checkpoint.pth.tar\"),model)\n",
        "\n",
        "    check_accuracy(val_loader,model,device=DEVICE)\n",
        "    scaler=torch.cuda.amp.GradScaler()\n",
        "    for epoch in range(NUM_EPOCHS):\n",
        "      train_fn(train_loader,model,optimizer,loss_fn,scaler)\n",
        "\n",
        "    #save model\n",
        "    checkpoint={\n",
        "        \"state_dict\":model.state_dict(),\n",
        "        \"optimizer\":optimizer.state_dict()\n",
        "    }\n",
        "\n",
        "    save_checkpoint(checkpoint)\n",
        "\n",
        "    #check accuracy\n",
        "    check_accuracy(val_loader,model,device=DEVICE)\n",
        "\n",
        "    #print some examples to a folder\n",
        "    save_predictions_as_imgs(\n",
        "        val_loader,model,folder=\"saved_images/\",device=DEVICE\n",
        "\n",
        "    )\n",
        "\n",
        "if __name__=\"main\":\n",
        "  main()\n"
      ],
      "metadata": {
        "id": "1pMXfvQedtmb"
      },
      "execution_count": null,
      "outputs": []
    }
  ]
}