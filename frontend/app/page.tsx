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
    senior_citizen: 0,
    tenure_months: 12,
    monthly_charge: 85,
    total_charges: 1200,
    gender: "Female",
    contract_type: "Month-to-month",
    payment_method: "Electronic check",
    internet_service: "Fiber optic",
    tech_support: "No",
    online_security: "No",
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

    try {
      const predictionResponse = await api.post("/predict", customer);
      const explanationResponse = await api.post("/explain", customer);
      const fairnessResponse = await api.get("/fairness");

      setResult(predictionResponse.data.result);
      setExplanations(explanationResponse.data.explanation.top_factors);
      setFairnessGroups(fairnessResponse.data.fairness_report.group_metrics);
      setFairnessSummary(
        fairnessResponse.data.fairness_report.fairness_summary
      );
    } finally {
      setLoading(false);
    }
  };

  return (
    <main className="min-h-screen bg-slate-950 text-white p-10">
      <div className="max-w-6xl mx-auto space-y-8">
        <div>
          <h1 className="text-5xl font-bold">
            SaaS Customer Retention Intelligence
          </h1>
          <p className="text-slate-400 mt-3 text-lg">
            Trustworthy churn prediction with explainable AI and fairness
            analysis using real-world customer churn data.
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
          <h2 className="text-2xl font-semibold mb-2">
            Customer Retention Risk Input
          </h2>

          <p className="text-slate-400 mb-5">
            Enter a customer profile to estimate churn risk and understand the
            factors influencing the prediction.
          </p>

          <div className="grid grid-cols-1 md:grid-cols-2 gap-5">

            <div>
              <label className="block text-sm text-slate-400 mb-2">
                Senior Citizen
              </label>

              <select
                className="w-full bg-slate-950 border border-slate-800 rounded-xl p-3"
                value={customer.senior_citizen}
                onChange={(e) =>
                  updateCustomer("senior_citizen", Number(e.target.value))
                }
              >
                <option value={0}>No</option>
                <option value={1}>Yes</option>
              </select>
            </div>

            <div>
              <label className="block text-sm text-slate-400 mb-2">
                Tenure (Months)
              </label>

              <input
                className="w-full bg-slate-950 border border-slate-800 rounded-xl p-3"
                type="number"
                value={customer.tenure_months}
                onChange={(e) =>
                  updateCustomer("tenure_months", Number(e.target.value))
                }
                placeholder="Enter customer tenure"
              />
            </div>

            <div>
              <label className="block text-sm text-slate-400 mb-2">
                Monthly Charge ($)
              </label>

              <input
                className="w-full bg-slate-950 border border-slate-800 rounded-xl p-3"
                type="number"
                value={customer.monthly_charge}
                onChange={(e) =>
                  updateCustomer("monthly_charge", Number(e.target.value))
                }
                placeholder="Enter monthly bill amount"
              />
            </div>

            <div>
              <label className="block text-sm text-slate-400 mb-2">
                Total Charges ($)
              </label>

              <input
                className="w-full bg-slate-950 border border-slate-800 rounded-xl p-3"
                type="number"
                value={customer.total_charges}
                onChange={(e) =>
                  updateCustomer("total_charges", Number(e.target.value))
                }
                placeholder="Enter total customer charges"
              />
            </div>

            <div>
              <label className="block text-sm text-slate-400 mb-2">
                Gender
              </label>

              <select
                className="w-full bg-slate-950 border border-slate-800 rounded-xl p-3"
                value={customer.gender}
                onChange={(e) => updateCustomer("gender", e.target.value)}
              >
                <option>Female</option>
                <option>Male</option>
              </select>
            </div>

            <div>
              <label className="block text-sm text-slate-400 mb-2">
                Contract Type
              </label>

              <select
                className="w-full bg-slate-950 border border-slate-800 rounded-xl p-3"
                value={customer.contract_type}
                onChange={(e) =>
                  updateCustomer("contract_type", e.target.value)
                }
              >
                <option>Month-to-month</option>
                <option>One year</option>
                <option>Two year</option>
              </select>
            </div>

            <div>
              <label className="block text-sm text-slate-400 mb-2">
                Payment Method
              </label>

              <select
                className="w-full bg-slate-950 border border-slate-800 rounded-xl p-3"
                value={customer.payment_method}
                onChange={(e) =>
                  updateCustomer("payment_method", e.target.value)
                }
              >
                <option>Electronic check</option>
                <option>Mailed check</option>
                <option>Bank transfer (automatic)</option>
                <option>Credit card (automatic)</option>
              </select>
            </div>

            <div>
              <label className="block text-sm text-slate-400 mb-2">
                Internet Service
              </label>

              <select
                className="w-full bg-slate-950 border border-slate-800 rounded-xl p-3"
                value={customer.internet_service}
                onChange={(e) =>
                  updateCustomer("internet_service", e.target.value)
                }
              >
                <option>Fiber optic</option>
                <option>DSL</option>
                <option>No</option>
              </select>
            </div>

            <div>
              <label className="block text-sm text-slate-400 mb-2">
                Tech Support
              </label>

              <select
                className="w-full bg-slate-950 border border-slate-800 rounded-xl p-3"
                value={customer.tech_support}
                onChange={(e) =>
                  updateCustomer("tech_support", e.target.value)
                }
              >
                <option>No</option>
                <option>Yes</option>
                <option>No internet service</option>
              </select>
            </div>

            <div>
              <label className="block text-sm text-slate-400 mb-2">
                Online Security
              </label>

              <select
                className="w-full bg-slate-950 border border-slate-800 rounded-xl p-3"
                value={customer.online_security}
                onChange={(e) =>
                  updateCustomer("online_security", e.target.value)
                }
              >
                <option>No</option>
                <option>Yes</option>
                <option>No internet service</option>
              </select>
            </div>

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
                <p className="text-slate-400 text-sm">Probability</p>

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

           <div className="mt-10 h-[400px] bg-slate-950 rounded-xl p-5">
  <ResponsiveContainer width="100%" height="100%">
    <BarChart data={explanations}>
      <XAxis
        dataKey="feature"
        tick={{ fill: "#94a3b8", fontSize: 12 }}
        interval={0}
        angle={-25}
        textAnchor="end"
        height={90}
      />
      <YAxis tick={{ fill: "#94a3b8", fontSize: 12 }} />
      <Tooltip
        contentStyle={{
          backgroundColor: "#020617",
          border: "1px solid #334155",
          borderRadius: "12px",
          color: "#ffffff",
        }}
        labelStyle={{ color: "#ffffff" }}
        itemStyle={{ color: "#ffffff" }}
      />
      <Bar dataKey="shap_value" fill="#3b82f6" radius={[8, 8, 0, 0]} />
    </BarChart>
  </ResponsiveContainer>
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
          </section>
        )}
      </div>
    </main>
  );
}