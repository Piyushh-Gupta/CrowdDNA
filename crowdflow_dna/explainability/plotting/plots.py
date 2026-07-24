"""
Explainability Plotters.

Generates visual plots from ExplanationGraphs.
"""
import os
import matplotlib.pyplot as plt
from typing import List
import numpy as np

from crowdflow_dna.explainability.plotting.registry import PlotterProtocol, PlottingRegistry
from crowdflow_dna.explainability.session import ExplanationSession
from crowdflow_dna.explainability.models import ExplanationGraph

@PlottingRegistry.register("node_importance")
class NodeImportancePlotter(PlotterProtocol):
    
    def plot(self, session: ExplanationSession, graphs: List[ExplanationGraph]) -> None:
        plots_dir = os.path.join(session.output_directory, "plots")
        os.makedirs(plots_dir, exist_ok=True)
        
        for graph in graphs:
            if graph.node_importance is not None:
                importance = graph.node_importance.detach().cpu().numpy()
                nodes = np.arange(len(importance))
                
                plt.figure(figsize=(10, 6))
                plt.bar(nodes, importance, color="darkred")
                plt.title(f"Node Importance - {graph.sequence_id} ({graph.explainer_name})")
                plt.xlabel("Node Index")
                plt.ylabel("Importance Score (Probability Drop)")
                plt.grid(axis="y", alpha=0.3)
                
                plot_path = os.path.join(plots_dir, f"node_importance_{graph.sequence_id}_{graph.explainer_name}.png")
                plt.savefig(plot_path, bbox_inches="tight")
                plt.close()

@PlottingRegistry.register("confidence")
class ConfidencePlotter(PlotterProtocol):
    
    def plot(self, session: ExplanationSession, graphs: List[ExplanationGraph]) -> None:
        plots_dir = os.path.join(session.output_directory, "plots")
        os.makedirs(plots_dir, exist_ok=True)
        
        entropies = []
        margins = []
        labels = []
        
        for graph in graphs:
            if graph.confidence_analysis:
                entropies.append(graph.confidence_analysis.get("entropy", 0.0))
                margins.append(graph.confidence_analysis.get("margin", 0.0))
                labels.append(graph.sequence_id)
                
        if entropies:
            x = np.arange(len(labels))
            width = 0.35
            
            fig, ax = plt.subplots(figsize=(12, 6))
            ax.bar(x - width/2, entropies, width, label='Entropy', color="blue")
            ax.bar(x + width/2, margins, width, label='Margin', color="green")
            
            ax.set_ylabel('Scores')
            ax.set_title('Prediction Confidence Analysis')
            ax.set_xticks(x)
            ax.set_xticklabels(labels, rotation=45, ha="right")
            ax.legend()
            
            plt.grid(axis="y", alpha=0.3)
            plot_path = os.path.join(plots_dir, "confidence_analysis.png")
            plt.savefig(plot_path, bbox_inches="tight")
            plt.close()
