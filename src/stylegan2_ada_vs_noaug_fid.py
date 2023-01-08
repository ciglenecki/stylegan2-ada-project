import matplotlib.pyplot as plt
import numpy as np
import json

ada_ = "ada-fid50k.jsonl"
noaug_ = "noaug-fid50k.jsonl"


def get_jsonl(filename):
    with open(filename, "r") as json_file:
        json_list = list(json_file)

    results = []
    for json_str in json_list:
        results.append(json.loads(json_str))
    return results


ada_json = get_jsonl(ada_)
noaug_json = get_jsonl(noaug_)

print(ada_json)
adas = [j["results"]["fid50k_full"] for j in ada_json]
noaugs = [j["results"]["fid50k_full"] for j in noaug_json]

min_ = min(len(adas), len(noaugs))
adas = adas[:min_]
noaugs = noaugs[:min_]

x = np.linspace(0, 56, min_)
plt.ylabel("FID (Fréchet inception distance)")
plt.xlabel("kimg [thousands of images]")

plt.title(
    "ADA vs. no augmentation\nPretrained StyleGAN2 finetuned on MetFaces images (n = 100)"
)
plt.plot(x, adas, label="ADA")
plt.plot(x, noaugs, label="No augmentation")
plt.legend()
plt.show()
