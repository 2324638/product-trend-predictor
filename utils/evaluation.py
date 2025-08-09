import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, mean_absolute_percentage_error
from typing import Dict, List, Tuple, Any
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots


class ModelEvaluator:
    """
    Comprehensive model evaluation utilities for trend prediction models
    """
    
    def __init__(self):
        self.metrics_history = []
        
    def calculate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray, model_name: str = "Model") -> Dict[str, float]:
        """Calculate comprehensive evaluation metrics"""
        
        # Ensure arrays are 1D
        y_true = np.array(y_true).flatten()
        y_pred = np.array(y_pred).flatten()
        
        # Remove any NaN values
        mask = ~(np.isnan(y_true) | np.isnan(y_pred))
        y_true = y_true[mask]
        y_pred = y_pred[mask]
        
        if len(y_true) == 0:
            return {"error": "No valid predictions to evaluate"}
        
        metrics = {
            "model_name": model_name,
            "mae": mean_absolute_error(y_true, y_pred),
            "mse": mean_squared_error(y_true, y_pred),
            "rmse": np.sqrt(mean_squared_error(y_true, y_pred)),
            "r2_score": r2_score(y_true, y_pred),
            "mape": mean_absolute_percentage_error(y_true, y_pred) * 100,
            "n_samples": len(y_true)
        }
        
        # Additional custom metrics
        metrics["normalized_rmse"] = metrics["rmse"] / (np.max(y_true) - np.min(y_true))
        metrics["mean_bias"] = np.mean(y_pred - y_true)
        metrics["directional_accuracy"] = self.calculate_directional_accuracy(y_true, y_pred)
        
        # Trend prediction specific metrics
        if len(y_true) > 1:
            metrics["trend_accuracy"] = self.calculate_trend_accuracy(y_true, y_pred)
        
        return metrics
    
    def calculate_directional_accuracy(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate the percentage of correct directional predictions"""
        if len(y_true) < 2:
            return 0.0
            
        true_directions = np.diff(y_true) > 0
        pred_directions = np.diff(y_pred) > 0
        
        return np.mean(true_directions == pred_directions) * 100
    
    def calculate_trend_accuracy(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate accuracy of trend prediction (up/down/stable)"""
        if len(y_true) < 2:
            return 0.0
            
        def categorize_trend(values):
            if len(values) < 2:
                return 'stable'
            slope = np.polyfit(range(len(values)), values, 1)[0]
            threshold = np.std(values) * 0.1  # Dynamic threshold
            
            if slope > threshold:
                return 'up'
            elif slope < -threshold:
                return 'down'
            else:
                return 'stable'
        
        true_trend = categorize_trend(y_true)
        pred_trend = categorize_trend(y_pred)
        
        return 100.0 if true_trend == pred_trend else 0.0
    
    def evaluate_ensemble(self, y_true: np.ndarray, predictions: Dict[str, np.ndarray]) -> Dict[str, Any]:
        """Evaluate multiple models and ensemble performance"""
        results = {}
        
        # Evaluate individual models
        for model_name, y_pred in predictions.items():
            results[model_name] = self.calculate_metrics(y_true, y_pred, model_name)
        
        # Calculate ensemble prediction (simple average)
        if len(predictions) > 1:
            ensemble_pred = np.mean(list(predictions.values()), axis=0)
            results["ensemble"] = self.calculate_metrics(y_true, ensemble_pred, "Ensemble")
        
        # Model ranking
        if len(results) > 1:
            ranking = sorted(
                [(name, metrics.get("r2_score", 0)) for name, metrics in results.items() if isinstance(metrics, dict) and "r2_score" in metrics],
                key=lambda x: x[1],
                reverse=True
            )
            results["model_ranking"] = ranking
        
        return results
    
    def create_evaluation_plots(self, y_true: np.ndarray, predictions: Dict[str, np.ndarray], 
                              save_path: str = None) -> Dict[str, Any]:
        """Create comprehensive evaluation plots"""
        
        # Set up the plotting style
        plt.style.use('seaborn-v0_8')
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Model Evaluation Dashboard', fontsize=16, fontweight='bold')
        
        # 1. Actual vs Predicted scatter plot
        ax1 = axes[0, 0]
        colors = plt.cm.tab10(np.linspace(0, 1, len(predictions)))
        
        for i, (model_name, y_pred) in enumerate(predictions.items()):
            ax1.scatter(y_true, y_pred, alpha=0.6, label=model_name, color=colors[i])
        
        # Perfect prediction line
        min_val, max_val = min(y_true.min(), min([p.min() for p in predictions.values()])), \
                          max(y_true.max(), max([p.max() for p in predictions.values()]))
        ax1.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.8, linewidth=2)
        
        ax1.set_xlabel('Actual Values')
        ax1.set_ylabel('Predicted Values')
        ax1.set_title('Actual vs Predicted')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. Residuals plot
        ax2 = axes[0, 1]
        for i, (model_name, y_pred) in enumerate(predictions.items()):
            residuals = y_pred - y_true
            ax2.scatter(y_pred, residuals, alpha=0.6, label=model_name, color=colors[i])
        
        ax2.axhline(y=0, color='r', linestyle='--', alpha=0.8)
        ax2.set_xlabel('Predicted Values')
        ax2.set_ylabel('Residuals')
        ax2.set_title('Residuals Plot')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. Time series comparison
        ax3 = axes[1, 0]
        x_axis = range(len(y_true))
        ax3.plot(x_axis, y_true, 'black', linewidth=2, label='Actual', marker='o', markersize=4)
        
        for i, (model_name, y_pred) in enumerate(predictions.items()):
            ax3.plot(x_axis, y_pred, color=colors[i], linewidth=1.5, 
                    label=f'{model_name}', alpha=0.8, linestyle='--')
        
        ax3.set_xlabel('Time Index')
        ax3.set_ylabel('Values')
        ax3.set_title('Time Series Comparison')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 4. Metrics comparison bar chart
        ax4 = axes[1, 1]
        metrics_data = {}
        for model_name, y_pred in predictions.items():
            metrics = self.calculate_metrics(y_true, y_pred, model_name)
            metrics_data[model_name] = metrics.get('r2_score', 0)
        
        models = list(metrics_data.keys())
        r2_scores = list(metrics_data.values())
        bars = ax4.bar(models, r2_scores, color=colors[:len(models)], alpha=0.7)
        
        # Add value labels on bars
        for bar, score in zip(bars, r2_scores):
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{score:.3f}', ha='center', va='bottom')
        
        ax4.set_ylabel('R² Score')
        ax4.set_title('Model Performance Comparison')
        ax4.set_ylim(0, 1)
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def create_interactive_evaluation_dashboard(self, y_true: np.ndarray, 
                                              predictions: Dict[str, np.ndarray],
                                              dates: List[str] = None) -> go.Figure:
        """Create an interactive Plotly dashboard for model evaluation"""
        
        if dates is None:
            dates = [f"Day {i+1}" for i in range(len(y_true))]
        
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Actual vs Predicted', 'Time Series Comparison', 
                          'Model Metrics', 'Residuals Distribution'),
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        colors = px.colors.qualitative.Set1
        
        # 1. Actual vs Predicted scatter
        for i, (model_name, y_pred) in enumerate(predictions.items()):
            fig.add_trace(
                go.Scatter(
                    x=y_true, y=y_pred,
                    mode='markers',
                    name=f'{model_name}',
                    marker=dict(color=colors[i % len(colors)], size=6, opacity=0.7),
                    showlegend=True
                ),
                row=1, col=1
            )
        
        # Perfect prediction line
        min_val = min(y_true.min(), min([p.min() for p in predictions.values()]))
        max_val = max(y_true.max(), max([p.max() for p in predictions.values()]))
        fig.add_trace(
            go.Scatter(
                x=[min_val, max_val], y=[min_val, max_val],
                mode='lines',
                name='Perfect Prediction',
                line=dict(color='red', dash='dash', width=2),
                showlegend=False
            ),
            row=1, col=1
        )
        
        # 2. Time series comparison
        fig.add_trace(
            go.Scatter(
                x=dates, y=y_true,
                mode='lines+markers',
                name='Actual',
                line=dict(color='black', width=3),
                marker=dict(size=6),
                showlegend=False
            ),
            row=1, col=2
        )
        
        for i, (model_name, y_pred) in enumerate(predictions.items()):
            fig.add_trace(
                go.Scatter(
                    x=dates, y=y_pred,
                    mode='lines',
                    name=f'{model_name} Pred',
                    line=dict(color=colors[i % len(colors)], width=2, dash='dash'),
                    showlegend=False
                ),
                row=1, col=2
            )
        
        # 3. Model metrics comparison
        metrics_names = []
        r2_scores = []
        for model_name, y_pred in predictions.items():
            metrics = self.calculate_metrics(y_true, y_pred, model_name)
            metrics_names.append(model_name)
            r2_scores.append(metrics.get('r2_score', 0))
        
        fig.add_trace(
            go.Bar(
                x=metrics_names, y=r2_scores,
                name='R² Score',
                marker=dict(color=colors[:len(metrics_names)]),
                showlegend=False,
                text=[f'{score:.3f}' for score in r2_scores],
                textposition='auto'
            ),
            row=2, col=1
        )
        
        # 4. Residuals distribution
        for i, (model_name, y_pred) in enumerate(predictions.items()):
            residuals = y_pred - y_true
            fig.add_trace(
                go.Histogram(
                    x=residuals,
                    name=f'{model_name} Residuals',
                    opacity=0.7,
                    marker=dict(color=colors[i % len(colors)]),
                    showlegend=False
                ),
                row=2, col=2
            )
        
        # Update layout
        fig.update_layout(
            height=800,
            title_text="Interactive Model Evaluation Dashboard",
            title_x=0.5,
            showlegend=True
        )
        
        # Update axis labels
        fig.update_xaxes(title_text="Actual Values", row=1, col=1)
        fig.update_yaxes(title_text="Predicted Values", row=1, col=1)
        fig.update_xaxes(title_text="Time", row=1, col=2)
        fig.update_yaxes(title_text="Values", row=1, col=2)
        fig.update_xaxes(title_text="Model", row=2, col=1)
        fig.update_yaxes(title_text="R² Score", row=2, col=1)
        fig.update_xaxes(title_text="Residuals", row=2, col=2)
        fig.update_yaxes(title_text="Frequency", row=2, col=2)
        
        return fig
    
    def generate_evaluation_report(self, y_true: np.ndarray, 
                                 predictions: Dict[str, np.ndarray],
                                 model_descriptions: Dict[str, str] = None) -> str:
        """Generate a comprehensive evaluation report"""
        
        report = []
        report.append("=" * 80)
        report.append("MODEL EVALUATION REPORT")
        report.append("=" * 80)
        report.append(f"Generated on: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append(f"Number of samples: {len(y_true)}")
        report.append("")
        
        # Evaluate all models
        results = self.evaluate_ensemble(y_true, predictions)
        
        # Model performance summary
        report.append("MODEL PERFORMANCE SUMMARY")
        report.append("-" * 40)
        
        metrics_table = []
        for model_name, metrics in results.items():
            if isinstance(metrics, dict) and "mae" in metrics:
                metrics_table.append([
                    model_name,
                    f"{metrics['mae']:.4f}",
                    f"{metrics['rmse']:.4f}",
                    f"{metrics['r2_score']:.4f}",
                    f"{metrics['mape']:.2f}%",
                    f"{metrics['directional_accuracy']:.1f}%"
                ])
        
        # Format table
        headers = ["Model", "MAE", "RMSE", "R²", "MAPE", "Dir. Acc."]
        col_widths = [max(len(str(item)) for item in col) for col in zip(headers, *metrics_table)]
        
        header_row = " | ".join(h.ljust(w) for h, w in zip(headers, col_widths))
        report.append(header_row)
        report.append("-" * len(header_row))
        
        for row in metrics_table:
            report.append(" | ".join(str(item).ljust(w) for item, w in zip(row, col_widths)))
        
        report.append("")
        
        # Best model identification
        if "model_ranking" in results:
            report.append("MODEL RANKING")
            report.append("-" * 20)
            for i, (model_name, score) in enumerate(results["model_ranking"], 1):
                report.append(f"{i}. {model_name}: {score:.4f} R²")
            report.append("")
        
        # Detailed analysis for each model
        report.append("DETAILED MODEL ANALYSIS")
        report.append("-" * 30)
        
        for model_name, metrics in results.items():
            if isinstance(metrics, dict) and "mae" in metrics:
                report.append(f"\n{model_name.upper()}:")
                if model_descriptions and model_name in model_descriptions:
                    report.append(f"  Description: {model_descriptions[model_name]}")
                
                report.append(f"  • Mean Absolute Error: {metrics['mae']:.4f}")
                report.append(f"  • Root Mean Square Error: {metrics['rmse']:.4f}")
                report.append(f"  • R² Score: {metrics['r2_score']:.4f}")
                report.append(f"  • Mean Absolute Percentage Error: {metrics['mape']:.2f}%")
                report.append(f"  • Directional Accuracy: {metrics['directional_accuracy']:.1f}%")
                report.append(f"  • Mean Bias: {metrics['mean_bias']:.4f}")
                report.append(f"  • Normalized RMSE: {metrics['normalized_rmse']:.4f}")
                
                if "trend_accuracy" in metrics:
                    report.append(f"  • Trend Accuracy: {metrics['trend_accuracy']:.1f}%")
        
        # Recommendations
        report.append("\n" + "=" * 80)
        report.append("RECOMMENDATIONS")
        report.append("=" * 80)
        
        best_model = None
        if "model_ranking" in results and results["model_ranking"]:
            best_model = results["model_ranking"][0][0]
            best_r2 = results["model_ranking"][0][1]
            
            report.append(f"🏆 Best performing model: {best_model} (R² = {best_r2:.4f})")
            
            if best_r2 > 0.8:
                report.append("✅ Excellent model performance - ready for production")
            elif best_r2 > 0.6:
                report.append("✅ Good model performance - consider minor improvements")
            elif best_r2 > 0.4:
                report.append("⚠️  Moderate performance - consider feature engineering or model tuning")
            else:
                report.append("❌ Poor performance - requires significant improvements")
        
        # General recommendations
        report.append("\n📊 General Recommendations:")
        avg_mape = np.mean([metrics.get('mape', 100) for metrics in results.values() if isinstance(metrics, dict)])
        
        if avg_mape < 10:
            report.append("• Excellent prediction accuracy")
        elif avg_mape < 20:
            report.append("• Good prediction accuracy")
        else:
            report.append("• Consider improving data quality or model complexity")
        
        report.append("• Monitor model performance over time")
        report.append("• Retrain models regularly with new data")
        report.append("• Consider ensemble methods for improved robustness")
        
        return "\n".join(report)
    
    def save_evaluation_results(self, y_true: np.ndarray, predictions: Dict[str, np.ndarray],
                              output_dir: str = "evaluation_results") -> Dict[str, str]:
        """Save all evaluation results to files"""
        import os
        from datetime import datetime
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        saved_files = {}
        
        # Save metrics to CSV
        results = self.evaluate_ensemble(y_true, predictions)
        metrics_df = pd.DataFrame([
            metrics for metrics in results.values() 
            if isinstance(metrics, dict) and "mae" in metrics
        ])
        
        metrics_file = os.path.join(output_dir, f"model_metrics_{timestamp}.csv")
        metrics_df.to_csv(metrics_file, index=False)
        saved_files["metrics"] = metrics_file
        
        # Save evaluation plots
        plots_file = os.path.join(output_dir, f"evaluation_plots_{timestamp}.png")
        self.create_evaluation_plots(y_true, predictions, plots_file)
        saved_files["plots"] = plots_file
        
        # Save evaluation report
        report = self.generate_evaluation_report(y_true, predictions)
        report_file = os.path.join(output_dir, f"evaluation_report_{timestamp}.txt")
        with open(report_file, 'w') as f:
            f.write(report)
        saved_files["report"] = report_file
        
        # Save predictions data
        pred_data = {"actual": y_true}
        pred_data.update(predictions)
        pred_df = pd.DataFrame(pred_data)
        
        predictions_file = os.path.join(output_dir, f"predictions_{timestamp}.csv")
        pred_df.to_csv(predictions_file, index=False)
        saved_files["predictions"] = predictions_file
        
        return saved_files


# Example usage
if __name__ == "__main__":
    # Generate sample data for testing
    np.random.seed(42)
    n_samples = 100
    
    # True values with trend and noise
    x = np.linspace(0, 10, n_samples)
    y_true = 2 * x + np.sin(x) + np.random.normal(0, 0.5, n_samples)
    
    # Simulated predictions from different models
    predictions = {
        "Linear Model": y_true + np.random.normal(0, 0.3, n_samples),
        "Neural Network": y_true + np.random.normal(0, 0.4, n_samples),
        "Random Forest": y_true + np.random.normal(0, 0.5, n_samples)
    }
    
    # Create evaluator and test
    evaluator = ModelEvaluator()
    
    # Generate evaluation report
    report = evaluator.generate_evaluation_report(y_true, predictions)
    print(report)
    
    # Save results
    saved_files = evaluator.save_evaluation_results(y_true, predictions)
    print(f"\nEvaluation results saved to: {saved_files}")