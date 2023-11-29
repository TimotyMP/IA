{
  "nbformat": 4,
  "nbformat_minor": 0,
  "metadata": {
    "colab": {
      "provenance": [],
      "toc_visible": true,
      "authorship_tag": "ABX9TyPKsvJDLSqT4cKxntiLc0wK",
      "include_colab_link": true
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
      "cell_type": "markdown",
      "metadata": {
        "id": "view-in-github",
        "colab_type": "text"
      },
      "source": [
        "<a href=\"https://colab.research.google.com/github/TimotyMP/IA/blob/main/Practica1.ipynb\" target=\"_parent\"><img src=\"https://colab.research.google.com/assets/colab-badge.svg\" alt=\"Open In Colab\"/></a>"
      ]
    },
    {
      "cell_type": "markdown",
      "source": [
        "# Ejecucion 1"
      ],
      "metadata": {
        "id": "_jil3SjFSpSt"
      }
    },
    {
      "cell_type": "markdown",
      "source": [
        "Cada modificacion hay que compilar"
      ],
      "metadata": {
        "id": "SWelyAfGWH2I"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "import tensorflow as tf\n",
        "import numpy as np"
      ],
      "metadata": {
        "id": "LKGo0t6iByxr"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "celcius = np.array([-15, -5, 0, 5, 15], dtype=float)\n",
        "fahrenheit = np.array([5, 23, 32, 41, 59], dtype=float)"
      ],
      "metadata": {
        "id": "Szou1ylyNCmy"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "capa = tf.keras.layers.Dense(units=1, input_shape=[1])\n",
        "modelo = tf.keras.Sequential([capa])\n",
        "oculta1 = tf.keras.layers.Dense(units=3, input_shape=[1])\n",
        "oculta2 = tf.keras.layers.Dense(units=3)\n",
        "salida = tf.keras.layers.Dense(units=1)\n",
        "modelo = tf.keras.Sequential([oculta1, oculta2, salida])"
      ],
      "metadata": {
        "id": "bjJfjzQgDKI3"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "modelo.compile(\n",
        "    optimizer=tf.keras.optimizers.Adam(0.1),\n",
        "    loss='mean_squared_error'\n",
        ")"
      ],
      "metadata": {
        "id": "Wt0gMxR-DjuG"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "print(\"Comenzando entrenamiento...\")\n",
        "historial=modelo.fit(celcius, fahrenheit, epochs=1000, verbose=False)\n",
        "print(\"modelo entrenado!!\")"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "2EEGo3GUD5_8",
        "outputId": "5eb8fee7-cd92-49bd-9d4c-844970c1c24d"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Comenzando entrenamiento...\n",
            "modelo entrenado!!\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "import matplotlib.pyplot as plt\n",
        "plt.xlabel(\"# Epoca\")\n",
        "plt.ylabel(\"Magnitud de pérdida\")\n",
        "plt.plot(historial.history[\"loss\"])"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 466
        },
        "id": "ur6KKIUCFLSk",
        "outputId": "aba48a4c-c912-4ec7-8719-82cc2130568a"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "[<matplotlib.lines.Line2D at 0x7b1a5eeb9ba0>]"
            ]
          },
          "metadata": {},
          "execution_count": 6
        },
        {
          "output_type": "display_data",
          "data": {
            "text/plain": [
              "<Figure size 640x480 with 1 Axes>"
            ],
            "image/png": "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go01NTVBK85n/vz5+vrrr7Vy5crjzhUXF8vtdispKcnveHJysoqLi802R4ch33nfuRPJzc3V73//+yBUH7gjgSgsbw8AQMQJOBDdddddcrlcWrBggVq2bHnCVauDaefOnRo9erTy8vIUGxsb0vc62uOPP66xY8eajz0ej9LS0ix5bxc9RAAAWCrgQFRYWKjVq1erffv2oajnOKtXr9aePXv8ZrDV1NRo2bJlevHFF/XJJ5+osrJSZWVlfr1EJSUlSklJkSSlpKRoxYoVfq/rm4Xma3OsmJgYv9W4rcTWHQAAWCvgMUQdO3bUf/7zn1DUckL9+vXT+vXrVVhYaH5lZmZqyJAh5vfR0dFavHix+ZytW7eqqKhIWVlZkqSsrCytX79ee/bsMdvk5eUpISFBHTt2tOyz1JcrillmAABYKeAeomeffVaPPPKInnnmGXXu3FnR0dF+54M9A6tRo0bq1KmT37H4+Hg1bdrUPD5s2DCNHTtWTZo0UUJCgh544AFlZWXpkksukSRdeeWV6tixo26//XZNmTJFxcXFeuKJJzRy5Miw9QKdjK+HiEAEAIA1Ag5Evtlc/fr18zseykHVP+ePf/yjnE6nBg0apIqKCuXk5GjWrFnm+aioKC1YsED333+/srKyFB8fr6FDh2rChAmW11ofLrbuAADAUgEHos8++ywUdQTk888/93scGxurmTNnaubMmXU+p3Xr1lq4cGGIKwuOKGftnUwvgQgAAEsEHIguv/zyUNSBo0QdHtlFDxEAANYIOBCtW7fuhMcdDodiY2OVnp5uy3E5ZxKXr4fozFhEHACAM17Agahbt24nXXsoOjpaN998s15++WVL1w06m7C5KwAA1gp42v27776rtm3b6pVXXjGnwb/yyitq166d5s2bpz//+c9asmSJnnjiiVDUGxHY3BUAAGsF3EM0adIkPf/888rJyTGPde7cWa1atdK4ceO0YsUKxcfH6ze/+Y2ee+65oBYbKcytO7hlBgCAJQLuIVq/fr1at2593PHWrVtr/fr1kmpvq+3evfv0q4tQUaxDBACApQIORO3bt9fkyZNVWVlpHquqqtLkyZPN7Ty+//774zZTRf1FsVI1AACWCviW2cyZM3X99derVatW6tKli6TaXqOamhotWLBAkrR9+3aNGDEiuJVGEBZmBADAWgEHoksvvVQ7duzQ3Llz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33X2eDwqKCjwu85lZWVavXq12WbJkiXyer3q1auXBZ/C/g4ePCin0//XX1RUlLxerySuc6gE67pmZWVp2bJlqqqqMtvk5eWpXbt2p3y7TBLT7sNp/vz5RkxMjDFnzhxj06ZNxvDhw42kpCS/WTg4ufvvv99ITEw0Pv/8c2P37t3m18GDB8029913n5Genm4sWbLEWLVqlZGVlWVkZWWZ533Twa+88kqjsLDQWLRokdG8eXOmg/+Mo2eZGQbXORhWrFhhuFwuY9KkScY333xjzJ0712jQoIHx+uuvm20mT55sJCUlGe+//76xbt06Y8CAASectty9e3ejoKDA+OKLL4y2bdtG/HTwow0dOtQ455xzzGn377zzjtGsWTPjkUceMdtwnU/Nvn37jDVr1hhr1qwxJBnTpk0z1qxZY3z77beGYQTnupaVlRnJycnG7bffbmzYsMGYP3++0aBBA6bdn+leeOEFIz093XC73cbFF19sfPXVV+Eu6Ywi6YRfr776qtnmp59+MkaMGGE0btzYaNCggXHDDTcYu3fv9nudf//738bVV19txMXFGc2aNTN+85vfGFVVVRZ/mjPLsYGI6xwcH374odGpUycjJibGaN++vfHKK6/4nfd6vca4ceOM5ORkIyYmxujXr5+xdetWvzY//PCDccsttxgNGzY0EhISjLvuusvYt2+flR/D1jwejzF69GgjPT3diI2NNc4991zjd7/7nd80bq7zqfnss89O+Hfy0KFDDcMI3nVdu3at0bt3byMmJsY455xzjMmTJ5927Q7DOGppTgAAgAjEGCIAABDxCEQAACDiEYgAAEDEIxABAICIRyACAAARj0AEAAAiHoEIAABEPAIRAACIeAQiAAAQ8QhEAGxv7969crvdOnDggKqqqhQfH2/uAF+Xp556Sg6H47iv9u3bW1Q1gDOJK9wFAMDPyc/PV9euXRUfH6+CggI1adJE6enpP/u8Cy+8UJ9++qnfMZeLv/YAHI8eIgC29+WXX+qyyy6TJH3xxRfm9z/H5XIpJSXF76tZs2bm+TZt2mjixIm65ZZbFB8fr3POOUczZ870e42ioiINGDBADRs2VEJCgn7961+rpKTEr82HH36onj17KjY2Vs2aNdMNN9xgnvvf//1fZWZmqlGjRkpJSdGtt96qPXv2nOqlABAiBCIAtlRUVKSkpCQlJSVp2rRpevnll5WUlKT//u//1nvvvaekpCSNGDHitN9n6tSp6tq1q9asWaPHHntMo0ePVl5eniTJ6/VqwIABKi0t1dKlS5WXl6ft27fr5ptvNp//0Ucf6YYbbtA111yjNWvWaPHixbr44ovN81VVVZo4caLWrl2r9957T//+97915513nnbdAIKL3e4B2FJ1dbW+++47eTweZWZmatWqVYqPj1e3bt300UcfKT09XQ0bNvTr8TnaU089pYkTJyouLs7v+G233abZs2dLqu0h6tChgz7++GPz/ODBg+XxeLRw4ULl5eXp6quv1o4dO5SWliZJ2rRpky688EKtWLFCPXv21KWXXqpzzz1Xr7/+er0+16pVq9SzZ0/t27dPDRs2PJVLAyAE6CECYEsul0tt2rTRli1b1LNnT3Xp0kXFxcVKTk5W37591aZNmzrDkE+7du1UWFjo9zVhwgS/NllZWcc93rx5syRp8+bNSktLM8OQJHXs2FFJSUlmm8LCQvXr16/OGlavXq3rrrtO6enpatSokS6//HJJ+tlB4QCsxehCALZ04YUX6ttvv1VVVZW8Xq8aNmyo6upqVVdXq2HDhmrdurU2btx40tdwu906//zzQ1rnsT1QRztw4IBycnKUk5OjuXPnqnnz5ioqKlJOTo4qKytDWheAwNBDBMCWFi5cqMLCQqWkpOj1119XYWGhOnXqpOnTp6uwsFALFy4Myvt89dVXxz3u0KGDJKlDhw7auXOndu7caZ7ftGmTysrK1LFjR0lSly5dtHjx4hO+9pYtW/TDDz9o8uTJ6tOnj9q3b8+AasCm6CECYEutW7dWcXGxSkpKNGDAADkcDm3cuFGDBg1Sy5Yt6/Ua1dXVKi4u9jvmcDiUnJxsPl6+fLmmTJmigQMHKi8vT2+99ZY++ugjSVJ2drY6d+6sIUOGaPr06aqurtaIESN0+eWXKzMzU5L05JNPql+/fjrvvPM0ePBgVVdXa+HChXr00UeVnp4ut9utF154Qffdd582bNigiRMnBukKAQgmeogA2Nbnn39uTmdfsWKFWrVqVe8wJEkbN25Uy5Yt/b5at27t1+Y3v/mNVq1ape7du+vpp5/WtGnTlJOTI6k2PL3//vtq3Lix+vbtq+zsbJ177rl68803zef/4he/0FtvvaUPPvhA3bp103/9139pxYoVkqTmzZtrzpw5euutt9SxY0dNnjxZzz33XBCuDIBgY5YZgIjVpk0bjRkzRmPGjAl3KQDCjB4iAAAQ8QhEAAAg4nHLDAAARDx6iAAAQMQjEAEAgIhHIAIAABGPQAQAACIegQgAAEQ8AhEAAIh4BCIAABDxCEQAACDi/X8qFTzta95LuQAAAABJRU5ErkJggg==\n"
          },
          "metadata": {}
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "print(\"Realizar una conversion!!\")\n",
        "resultado = modelo.predict([100.0])\n",
        "print(\"El resultado es \" + str(resultado)+\" F°\")"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "4hFeK2kxHNm7",
        "outputId": "61543b40-7824-4d5f-965f-e1496c5c5725"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Realizar una conversion!!\n",
            "1/1 [==============================] - 0s 103ms/step\n",
            "El resultado es [[211.99437]] F°\n"
          ]
        }
      ]
    },
    {
      "cell_type": "markdown",
      "source": [
        "# Practica 1 .parte\n",
        "Implemantar la práctica anteriorreciclando, para cambiar el tipo de prediccion donde realice la conversión diferente (km am, gal a litros, etc)."
      ],
      "metadata": {
        "id": "gnNSbXPoLiie"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "import tensorflow as tf\n",
        "import numpy as np"
      ],
      "metadata": {
        "id": "2BG4jZJHLoEY"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "litros = np.array([-5,-3,-1,1,3,5,8], dtype=float)\n",
        "gal = np.array([-1.32,-0.79,-0.26,0.26,0.79,1.32,2.11], dtype=float)"
      ],
      "metadata": {
        "id": "HbT4psrnNy2m"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "capa = tf.keras.layers.Dense(units=1, input_shape=[1])\n",
        "modelo = tf.keras.Sequential([capa])"
      ],
      "metadata": {
        "id": "F4ZDGn9ROt9t"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "modelo.compile(\n",
        "    optimizer=tf.keras.optimizers.Adam(0.1),\n",
        "    loss='mean_squared_error'\n",
        ")"
      ],
      "metadata": {
        "id": "Q4EZvngrO8yJ"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "print(\"Comenzando entrenamiento...\")\n",
        "historial=modelo.fit(litros, gal, epochs=1000, verbose=False)\n",
        "print(\"modelo entrenado!!\")"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "1pXbabCKPEk8",
        "outputId": "c4ab148e-3e34-433c-edf4-790448abd713"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Comenzando entrenamiento...\n",
            "modelo entrenado!!\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "import matplotlib.pyplot as plt\n",
        "plt.xlabel(\"# Epoca\")\n",
        "plt.ylabel(\"Magnitud de pérdida\")\n",
        "plt.plot(historial.history[\"loss\"])"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 466
        },
        "id": "BBySlCxjPT95",
        "outputId": "62f69cf5-5dcd-4282-95e2-bab49d8c3870"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "[<matplotlib.lines.Line2D at 0x7b1a5dec6470>]"
            ]
          },
          "metadata": {},
          "execution_count": 13
        },
        {
          "output_type": "display_data",
          "data": {
            "text/plain": [
              "<Figure size 640x480 with 1 Axes>"
            ],
            "image/png": "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QUgCAABwg5AEAADgBiEJAADADUKSj1m8eLHi4uIUFBSkxMREbd++3eyS6pS0tDT17t1bjRs3VvPmzTVs2DAdPHjQpc/58+c1ceJENWvWTCEhIRo+fLjy8/Nd+uTl5emWW25Ro0aN1Lx5cz322GMqLy+vzUOpU+bOnSuLxaKpU6c62zjP3nHs2DHdc889atasmRo2bKhu3bppx44dzu2GYWjWrFlq0aKFGjZsqOTkZH311Vcu+ygoKNDo0aMVGhqq8PBwjR8/Xt9//31tH4pPq6io0MyZM9WmTRs1bNhQ7dq105w5c1ye78W59twnn3yiW2+9VdHR0bJYLFq7dq3Ldm+d0127dql///4KCgpSTEyM5s2b550DMOAzVq1aZQQEBBhLly419u7da0yYMMEIDw838vPzzS6tzkhJSTGWLVtm7Nmzx8jNzTWGDBlixMbGGt9//72zz4MPPmjExMQYmZmZxo4dO4y+ffsa/fr1c24vLy83unbtaiQnJxs7d+401q9fb0RERBgzZsww45B83vbt2424uDije/fuxpQpU5ztnOcrV1BQYLRu3dq49957jW3bthmHDh0yPvroI+Prr7929pk7d64RFhZmrF271vjiiy+M2267zWjTpo3xww8/OPsMGjTI6NGjh7F161bjn//8p9G+fXtj1KhRZhySz3r++eeNZs2aGevWrTMOHz5srFmzxggJCTH+8pe/OPtwrj23fv16449//KPxzjvvGJKMd99912W7N85pUVGRERkZaYwePdrYs2eP8fbbbxsNGzY0XnvttSuun5DkQ/r06WNMnDjR+XdFRYURHR1tpKWlmVhV3Xby5ElDkrF582bDMAyjsLDQaNCggbFmzRpnn/379xuSjKysLMMwLvxLbbVaDZvN5uzz6quvGqGhoUZJSUntHoCPO3v2rNGhQwcjIyPDGDBggDMkcZ6944knnjBuuOGGS253OBxGVFSUMX/+fGdbYWGhERgYaLz99tuGYRjGvn37DEnG559/7uzzj3/8w7BYLMaxY8dqrvg65pZbbjHuu+8+l7Y77rjDGD16tGEYnGtv+HlI8tY5feWVV4wmTZq4/HfjiSeeMDp27HjFNXO5zUeUlpYqOztbycnJzjar1ark5GRlZWWZWFndVlRUJElq2rSpJCk7O1tlZWUu57lTp06KjY11nuesrCx169ZNkZGR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\n"
          },
          "metadata": {}
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "print(\"Realizar una conversion!!\")\n",
        "resultado = modelo.predict([10])\n",
        "print(\"El resultado es \" + str(resultado)+\" galones\")"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "HSjhRnu1QOWL",
        "outputId": "0f0484c0-42ad-4510-c004-d7aa1076aa44"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Realizar una conversion!!\n",
            "1/1 [==============================] - 0s 70ms/step\n",
            "El resultado es [[2.6373224]] galones\n"
          ]
        }
      ]
    }
  ]
}
