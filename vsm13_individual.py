import json

def compute_dimensions(means: dict) -> dict:
    m = means
    PDI = 35 * (m[7]  - m[2])  + 25 * (m[20] - m[23])
    IDV = 35 * (m[4]  - m[1])  + 35 * (m[9]  - m[6])
    MAS = 35 * (m[5]  - m[3])  + 25 * (m[8]  - m[10])
    UAI = 40 * (m[18] - m[15]) + 25 * (m[21] - m[24])
    LTO = 40 * (m[13] - m[14]) + 25 * (m[19] - m[22])
    IVR = 35 * (m[12] - m[11]) + 40 * (m[17] - m[16])
    return {"PDI": PDI, "IDV": IDV, "MAS": MAS, "UAI": UAI, "LTO": LTO, "IVR": IVR}

def compute_individual_dimensions(individual_responses: dict) -> dict:
    individual_scores = {}
    for agent_name, responses in individual_responses.items():
        m = {int(k): v for k, v in responses.items()}
        individual_scores[agent_name] = compute_dimensions(m)
    return individual_scores

def main():
    input_path = "results/malaysia_day0_vsm13_run5.json"

    with open(input_path, "r") as f:
        data = json.load(f)

    individual_scores = compute_individual_dimensions(data["individual_responses"])
    data["individual_dimension_scores"] = individual_scores

    with open(input_path, "w") as f:
        json.dump(data, f, indent=2)

    print("Done. Individual dimension scores added to", input_path)
    for agent, scores in individual_scores.items():
        print(f"\n{agent}:")
        for dim, score in scores.items():
            print(f"  {dim}: {score}")

if __name__ == "__main__":
    main()