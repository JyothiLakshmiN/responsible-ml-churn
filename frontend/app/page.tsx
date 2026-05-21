"use client";

import { useState, useEffect } from "react";
import { api } from "@/lib/api";
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  Tooltip,
  ResponsiveContainer,
} from "recharts";

type PredictionResult = {
  churn_prediction: number;
  churn_probability: number;
  risk_level: string;
};

type ShapFactor = {
  feature: string;
  input_value: number;
  shap_value: number;
  impact: string;
};

type FairnessGroup = {
  group: string;
  count: number;
  churn_rate: number;
  auc_roc: number;
  accuracy: number;
};

type FairnessSummary = {
  auc_difference: number;
  accuracy_difference: number;
  fairness_status: string;
};

export default function Home() {
  const [result, setResult] = useState<PredictionResult | null>(null);
  const [explanations, setExplanations] = useState<ShapFactor[]>([]);
  const [loading, setLoading] = useState(false);
  const [fairnessGroups, setFairnessGroups] = useState<FairnessGroup[]>([]);
  const [fairnessSummary, setFairnessSummary] =
    useState<FairnessSummary | null>(null);
  const [apiStatus, setApiStatus] = useState("Checking...");

  const [customer, setCustomer] = useState({
    age: 35,
    tenure_months: 8,
    monthly_charge: 95,
    num_products: 1,
    support_calls: 5,
    gender: "Female",
    contract_type: "Month-to-Month",
    payment_method: "Electronic Check",
  });

  useEffect(() => {
  const checkHealth = async () => {
    try {
      const response = await api.get("/health");

      if (response.data.status === "ok") {
        setApiStatus("Online");
      }
    } catch {
      setApiStatus("Offline");
    }
  };

  checkHealth();
}, []);

  const updateCustomer = (field: string, value: string | number) => {
    setCustomer((prev) => ({
      ...prev,
      [field]: value,
    }));
  };

  const predict = async () => {
    setLoading(true);

    const predictionResponse = await api.post("/predict", customer);
    const explanationResponse = await api.post("/explain", customer);
    const fairnessResponse = await api.get("/fairness");

    setResult(predictionResponse.data.result);
    setExplanations(explanationResponse.data.explanation.top_factors);
    setFairnessGroups(fairnessResponse.data.fairness_report.group_metrics);
    setFairnessSummary(fairnessResponse.data.fairness_report.fairness_summary);

    setLoading(false);
  };

  return (
    <main className="min-h-screen bg-slate-950 text-white p-10">
      <div className="max-w-6xl mx-auto space-y-8">
        <div>
          <h1 className="text-5xl font-bold">Responsible ML Churn</h1>
          <p className="text-slate-400 mt-3 text-lg">
            Trustworthy churn prediction with explainable AI and fairness
            analysis.
          </p>
        </div>
        <div className="grid grid-cols-1 md:grid-cols-4 gap-5">
          <div className="bg-slate-900 border border-slate-800 rounded-2xl p-5">
            <p className="text-slate-400 text-sm">Model</p>
            <p className="text-xl font-bold mt-2">XGBoost</p>
          </div>

          <div className="bg-slate-900 border border-slate-800 rounded-2xl p-5">
            <p className="text-slate-400 text-sm">Explainability</p>
            <p className="text-xl font-bold mt-2">SHAP Enabled</p>
          </div>

          <div className="bg-slate-900 border border-slate-800 rounded-2xl p-5">
            <p className="text-slate-400 text-sm">Fairness Check</p>
            <p className="text-xl font-bold mt-2">Active</p>
          </div>

          <div className="bg-slate-900 border border-slate-800 rounded-2xl p-5">
            <p className="text-slate-400 text-sm">API Status</p>
            <p className="text-xl font-bold mt-2">{apiStatus}</p>
          </div>
        </div>
        <section className="rounded-2xl bg-slate-900 p-6 border border-slate-800">
          <h2 className="text-2xl font-semibold mb-5">
            Customer Risk Input
          </h2>

          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <input
              className="bg-slate-950 border border-slate-800 rounded-xl p-3"
              type="number"
              value={customer.age}
              onChange={(e) => updateCustomer("age", Number(e.target.value))}
              placeholder="Age"
            />

            <input
              className="bg-slate-950 border border-slate-800 rounded-xl p-3"
              type="number"
              value={customer.tenure_months}
              onChange={(e) =>
                updateCustomer("tenure_months", Number(e.target.value))
              }
              placeholder="Tenure Months"
            />

            <input
              className="bg-slate-950 border border-slate-800 rounded-xl p-3"
              type="number"
              value={customer.monthly_charge}
              onChange={(e) =>
                updateCustomer("monthly_charge", Number(e.target.value))
              }
              placeholder="Monthly Charge"
            />

            <input
              className="bg-slate-950 border border-slate-800 rounded-xl p-3"
              type="number"
              value={customer.num_products}
              onChange={(e) =>
                updateCustomer("num_products", Number(e.target.value))
              }
              placeholder="Number of Products"
            />

            <input
              className="bg-slate-950 border border-slate-800 rounded-xl p-3"
              type="number"
              value={customer.support_calls}
              onChange={(e) =>
                updateCustomer("support_calls", Number(e.target.value))
              }
              placeholder="Support Calls"
            />

            <select
              className="bg-slate-950 border border-slate-800 rounded-xl p-3"
              value={customer.gender}
              onChange={(e) => updateCustomer("gender", e.target.value)}
            >
              <option>Female</option>
              <option>Male</option>
            </select>

            <select
              className="bg-slate-950 border border-slate-800 rounded-xl p-3"
              value={customer.contract_type}
              onChange={(e) =>
                updateCustomer("contract_type", e.target.value)
              }
            >
              <option>Month-to-Month</option>
              <option>One Year</option>
              <option>Two Year</option>
            </select>

            <select
              className="bg-slate-950 border border-slate-800 rounded-xl p-3"
              value={customer.payment_method}
              onChange={(e) =>
                updateCustomer("payment_method", e.target.value)
              }
            >
              <option>Electronic Check</option>
              <option>Credit Card</option>
              <option>Bank Transfer</option>
              <option>Mailed Check</option>
            </select>
          </div>

          <button
            onClick={predict}
            disabled={loading}
            className="mt-6 px-6 py-3 rounded-xl bg-blue-600 hover:bg-blue-700 disabled:opacity-60 transition font-medium"
          >
            {loading ? "Analyzing..." : "Predict Churn"}
          </button>
        </section>

        {result && (
          <section className="rounded-2xl bg-slate-900 p-6 border border-slate-800">
            <h2 className="text-2xl font-semibold mb-5">Prediction Result</h2>

            <div className="grid grid-cols-1 md:grid-cols-3 gap-5">
              <div className="bg-slate-950 p-5 rounded-xl">
                <p className="text-slate-400 text-sm">Prediction</p>
                <p className="text-3xl font-bold mt-2">
                  {result.churn_prediction === 1 ? "Will Churn" : "Will Stay"}
                </p>
              </div>

              <div className="bg-slate-950 p-5 rounded-xl">
                <p className="text-slate-400 text-sm">
                  Probability
                </p>

                <p className="text-3xl font-bold mt-2">
                  {(result.churn_probability * 100).toFixed(2)}%
                </p>

                <div className="w-full bg-slate-800 rounded-full h-3 mt-4">
                  <div
                    className={`h-3 rounded-full ${result.churn_probability > 0.7
                      ? "bg-red-500"
                      : result.churn_probability > 0.4
                        ? "bg-yellow-500"
                        : "bg-green-500"
                      }`}
                    style={{
                      width: `${result.churn_probability * 100}%`,
                    }}
                  />
                </div>
              </div>

              <div className="bg-slate-950 p-5 rounded-xl">
                <p className="text-slate-400 text-sm">Risk Level</p>
                <p className="text-3xl font-bold mt-2">{result.risk_level}</p>
              </div>
            </div>
          </section>
        )}

        {explanations.length > 0 && (
          <section className="rounded-2xl bg-slate-900 p-6 border border-slate-800">
            <h2 className="text-2xl font-semibold mb-5">
              AI Explanation (SHAP)
            </h2>

            <div className="space-y-4">
              {explanations.map((factor, index) => (
                <div
                  key={index}
                  className="bg-slate-950 rounded-xl p-5 border border-slate-800"
                >
                  <div className="flex justify-between items-center gap-6">
                    <div>
                      <p className="font-semibold text-lg">{factor.feature}</p>
                      <p className="text-slate-400 text-sm mt-1">
                        Input Value: {factor.input_value}
                      </p>
                    </div>

                    <div className="text-right">
                      <p
                        className={`font-bold text-lg ${factor.shap_value > 0
                          ? "text-red-400"
                          : "text-green-400"
                          }`}
                      >
                        {factor.shap_value > 0 ? "+" : ""}
                        {factor.shap_value}
                      </p>

                      <p className="text-sm text-slate-400 mt-1">
                        {factor.impact}
                      </p>
                    </div>
                  </div>
                </div>
              ))}
            </div>
          </section>
        )}

        {fairnessSummary && (
          <section className="rounded-2xl bg-slate-900 p-6 border border-slate-800">
            <h2 className="text-2xl font-semibold mb-5">Fairness Analysis</h2>

            <div className="grid grid-cols-1 md:grid-cols-3 gap-5 mb-6">
              <div className="bg-slate-950 p-5 rounded-xl">
                <p className="text-slate-400 text-sm">Fairness Status</p>
                <p className="text-3xl font-bold mt-2">
                  {fairnessSummary.fairness_status}
                </p>
              </div>

              <div className="bg-slate-950 p-5 rounded-xl">
                <p className="text-slate-400 text-sm">AUC Difference</p>
                <p className="text-3xl font-bold mt-2">
                  {fairnessSummary.auc_difference}
                </p>
              </div>

              <div className="bg-slate-950 p-5 rounded-xl">
                <p className="text-slate-400 text-sm">Accuracy Difference</p>
                <p className="text-3xl font-bold mt-2">
                  {fairnessSummary.accuracy_difference}
                </p>
              </div>
            </div>

            <div className="space-y-4">
              {fairnessGroups.map((group) => (
                <div
                  key={group.group}
                  className="bg-slate-950 rounded-xl p-5 border border-slate-800"
                >
                  <div className="flex justify-between">
                    <div>
                      <p className="text-xl font-semibold">{group.group}</p>
                      <p className="text-slate-400 text-sm">
                        Count: {group.count}
                      </p>
                    </div>

                    <div className="text-right">
                      <p>AUC: {group.auc_roc}</p>
                      <p>Accuracy: {(group.accuracy * 100).toFixed(2)}%</p>
                      <p>Churn Rate: {(group.churn_rate * 100).toFixed(2)}%</p>
                    </div>
                  </div>
                </div>
              ))}
            </div>
            <div className="mt-10 h-[400px] bg-slate-950 rounded-xl p-5">
              <ResponsiveContainer width="100%" height="100%">
                <BarChart data={explanations}>
                  <XAxis dataKey="feature" />
                  <YAxis />
                  <Tooltip />
                  <Bar dataKey="shap_value" />
                </BarChart>
              </ResponsiveContainer>
            </div>
          </section>
        )}
      </div>
    </main>
  );
}