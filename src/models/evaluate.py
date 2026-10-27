def evaluation_table(model_results, y_test, y_train):
    rows = []
    for name, result in model_results.items():
        row = {
            "model": name,
            "F1-best threshold (OOF)": result["threshold_f1"],
            "test net value at F1 threshold (£)": result["test_net_value_f1"],
        }
        for metric in [
            "accuracy",
            "precision",
            "recall",
            "f1",
            "roc_auc",
            "average_precision",
        ]:
            fold_values = result["fold"][f"test_{metric}"]
            row[f"CV {metric} mean"] = float(np.mean(fold_values))
            row[f"CV {metric} std"] = float(np.std(fold_values, ddof=1))
            test_key = "avg_precision" if metric == "average_precision" else metric
            row[f"test {test_key}"] = result["test_metrics"][test_key]
        rows.append(row)
    return pd.DataFrame(rows).set_index("model")
