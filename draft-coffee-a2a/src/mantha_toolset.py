"""
MANTHA — mantha_toolset.py
Wraps the pipeline tools (fetcher, transformer, categorizer, plotter,
report, mail) as async methods that the A2A OpenAI executor can call
via function-calling.
"""

import json
import logging
import os
import tempfile
from typing import Any, Optional

log = logging.getLogger("mantha_toolset")


class ManthaToolset:
    """MANTHA General Data Pipeline Toolset — fetch, transform, categorise, plot, report, send."""

    def __init__(self):
        # All tool modules live alongside this file in src/
        import sys, pathlib
        sys.path.insert(0, str(pathlib.Path(__file__).parent))

    # ── 1. Fetch ───────────────────────────────────────────────────────────────
    async def fetch_data(self, filepath: str, required_columns: str = "") -> str:
        """Load a CSV or Excel file and return a summary of its contents.

        Args:
            filepath: Path to the CSV or Excel file to load.
            required_columns: Comma-separated list of column names that must be present (optional).
        """
        try:
            from fetcher import fetch_data as _fetch
            cols = [c.strip() for c in required_columns.split(",") if c.strip()] or None
            df = _fetch(filepath, required_columns=cols)
            if df is None:
                return json.dumps({"status": "error", "message": f"Could not load file: {filepath}"})
            return json.dumps({
                "status": "ok",
                "rows": len(df),
                "columns": df.columns.tolist(),
                "preview": df.head(5).to_dict(orient="records"),
            }, default=str)
        except Exception as e:
            return json.dumps({"status": "error", "message": str(e)})

    # ── 2. Transform ───────────────────────────────────────────────────────────
    async def transform_data(self, filepath: str, use_llm: bool = True) -> str:
        """Clean, normalise and type-coerce a data file. Returns cleaned data summary.

        Args:
            filepath: Path to the CSV or Excel file to transform.
            use_llm: Whether to use the LLM to infer column types (default: true).
        """
        try:
            from fetcher import fetch_data as _fetch
            from transformer import transform_data as _transform
            df = _fetch(filepath)
            if df is None:
                return json.dumps({"status": "error", "message": "Could not load file."})
            df = _transform(df, use_llm=use_llm)
            if df is None:
                return json.dumps({"status": "error", "message": "Transformation failed."})
            return json.dumps({
                "status": "ok",
                "rows": len(df),
                "columns": df.columns.tolist(),
                "dtypes": {col: str(dt) for col, dt in df.dtypes.items()},
                "preview": df.head(5).to_dict(orient="records"),
            }, default=str)
        except Exception as e:
            return json.dumps({"status": "error", "message": str(e)})

    # ── 3. Categorise ──────────────────────────────────────────────────────────
    async def categorise_data(
        self,
        filepath: str,
        source_column: str,
        categories: str = "",
        context: str = "Categorise these data values meaningfully.",
    ) -> str:
        """Use the LLM to assign categories to a column in the data file.

        Args:
            filepath: Path to the CSV or Excel file.
            source_column: Name of the column whose values should be categorised.
            categories: Comma-separated list of allowed category names (optional — LLM chooses freely if empty).
            context: A hint describing what the data represents to guide the LLM.
        """
        try:
            from fetcher import fetch_data as _fetch
            from transformer import transform_data as _transform
            from categorizer import categorise_column, auto_breakdown

            df = _fetch(filepath)
            if df is None:
                return json.dumps({"status": "error", "message": "Could not load file."})
            df = _transform(df, use_llm=False)
            if df is None:
                return json.dumps({"status": "error", "message": "Transformation failed."})

            cat_list = [c.strip() for c in categories.split(",") if c.strip()] or None
            df = categorise_column(
                df,
                source_column=source_column,
                output_column="llm_category",
                categories=cat_list,
                context=context,
            )
            breakdown = auto_breakdown(df)
            return json.dumps({
                "status": "ok",
                "category_counts": df["llm_category"].value_counts().to_dict(),
                "breakdown": breakdown,
                "preview": df[[source_column, "llm_category"]].head(10).to_dict(orient="records"),
            }, default=str)
        except Exception as e:
            return json.dumps({"status": "error", "message": str(e)})

    # ── 4. Plot ────────────────────────────────────────────────────────────────
    async def plot_data(self, filepath: str, output_dir: str = "mantha_plots") -> str:
        """Generate charts and graphs from a data file. Returns paths to saved plot images.

        Args:
            filepath: Path to the CSV or Excel file to plot.
            output_dir: Directory where plot PNG files will be saved (default: mantha_plots).
        """
        try:
            from fetcher import fetch_data as _fetch
            from transformer import transform_data as _transform
            from categorizer import auto_breakdown
            from plotter import auto_plot

            df = _fetch(filepath)
            if df is None:
                return json.dumps({"status": "error", "message": "Could not load file."})
            df = _transform(df, use_llm=False)
            if df is None:
                return json.dumps({"status": "error", "message": "Transformation failed."})

            breakdown = auto_breakdown(df)
            plot_paths = auto_plot(df, breakdown=breakdown, output_dir=output_dir)
            return json.dumps({
                "status": "ok",
                "plots_generated": len(plot_paths),
                "plot_paths": plot_paths,
            })
        except Exception as e:
            return json.dumps({"status": "error", "message": str(e)})

    # ── 5. Generate Report ─────────────────────────────────────────────────────
    async def generate_report(
        self,
        filepath: str,
        output_pdf: str = "mantha_report.pdf",
        report_title: str = "MANTHA — Data Pipeline Report",
        summary_text: str = "",
    ) -> str:
        """Run the full pipeline (fetch → transform → categorise → plot) and produce a PDF report.

        Args:
            filepath: Path to the CSV or Excel file to process.
            output_pdf: Filename/path for the output PDF (default: mantha_report.pdf).
            report_title: Title to display on the report cover page.
            summary_text: Custom summary text to include. If empty, an automatic summary is generated.
        """
        try:
            from fetcher import fetch_data as _fetch
            from transformer import transform_data as _transform
            from categorizer import auto_breakdown
            from plotter import auto_plot
            from report import create_report

            df = _fetch(filepath)
            if df is None:
                return json.dumps({"status": "error", "message": "Could not load file."})
            df = _transform(df, use_llm=True)
            if df is None:
                return json.dumps({"status": "error", "message": "Transformation failed."})

            breakdown = auto_breakdown(df)
            plot_paths = auto_plot(df, breakdown=breakdown)

            auto_summary = (
                f"Pipeline run completed successfully.\n"
                f"Rows processed: {len(df):,}  |  Columns: {len(df.columns)}\n"
                f"Plots generated: {len(plot_paths)}\n"
                f"Breakdown: {breakdown}"
            )

            pdf_path = create_report(
                report_title=report_title,
                report_subtitle=f"Source: {filepath}",
                summary_text=summary_text or auto_summary,
                dataframes=[("Processed Data", df)],
                plot_paths=plot_paths,
                output_path=output_pdf,
            )
            if pdf_path is None:
                return json.dumps({"status": "error", "message": "PDF generation failed."})
            return json.dumps({
                "status": "ok",
                "pdf_path": pdf_path,
                "plots": plot_paths,
                "summary": auto_summary,
            })
        except Exception as e:
            return json.dumps({"status": "error", "message": str(e)})

    # ── 6. Send Report ─────────────────────────────────────────────────────────
    async def send_report(
        self,
        pdf_path: str,
        recipients: str,
        subject: str = "MANTHA — Pipeline Report Ready",
        summary: str = "Please find the latest MANTHA pipeline report attached.",
    ) -> str:
        """Email the generated PDF report to one or more recipients via Gmail.

        Args:
            pdf_path: Path to the PDF file to attach and send.
            recipients: Comma-separated list of recipient email addresses.
            subject: Email subject line.
            summary: Short summary text to include in the email body.
        """
        try:
            from mail import send_report as _send
            to_list = [r.strip() for r in recipients.split(",") if r.strip()]
            if not to_list:
                return json.dumps({"status": "error", "message": "No valid recipients provided."})
            sent = _send(
                to=to_list,
                subject=subject,
                summary=summary,
                attachment_paths=[pdf_path] if pdf_path else [],
            )
            return json.dumps({
                "status": "ok" if sent else "error",
                "sent": sent,
                "recipients": to_list,
            })
        except Exception as e:
            return json.dumps({"status": "error", "message": str(e)})

    # ── 7. Full Pipeline (convenience) ────────────────────────────────────────
    async def run_full_pipeline(
        self,
        filepath: str,
        recipients: str = "",
        source_column: str = "",
        report_title: str = "MANTHA — Data Pipeline Report",
    ) -> str:
        """Run the complete end-to-end pipeline: fetch → transform → categorise → plot → report → email.

        Args:
            filepath: Path to the CSV or Excel input file.
            recipients: Comma-separated email addresses to send the report to (optional).
            source_column: Column to LLM-categorise (optional — skipped if empty).
            report_title: Title for the PDF report cover page.
        """
        try:
            from fetcher import fetch_data as _fetch
            from transformer import transform_data as _transform
            from categorizer import categorise_column, auto_breakdown
            from plotter import auto_plot
            from report import create_report
            from mail import send_report as _send

            # Step 1 — Fetch
            df = _fetch(filepath)
            if df is None:
                return json.dumps({"status": "error", "step": "fetch", "message": "Could not load file."})

            # Step 2 — Transform
            df = _transform(df, use_llm=True)
            if df is None:
                return json.dumps({"status": "error", "step": "transform", "message": "Transformation failed."})

            # Step 3 — Categorise (optional)
            breakdown = auto_breakdown(df)
            if source_column and source_column in df.columns:
                df = categorise_column(df, source_column=source_column, output_column="llm_category")

            # Step 4 — Plot
            plot_paths = auto_plot(df, breakdown=breakdown)

            # Step 5 — Report
            summary = (
                f"Pipeline completed.\n"
                f"Rows: {len(df):,}  |  Columns: {len(df.columns)}\n"
                f"Plots: {len(plot_paths)}"
            )
            pdf_path = create_report(
                report_title=report_title,
                report_subtitle=f"Source: {filepath}",
                summary_text=summary,
                dataframes=[("Processed Data", df)],
                plot_paths=plot_paths,
                output_path="mantha_report.pdf",
            )
            if pdf_path is None:
                return json.dumps({"status": "error", "step": "report", "message": "PDF generation failed."})

            # Step 6 — Send (optional)
            email_sent = False
            to_list = [r.strip() for r in recipients.split(",") if r.strip()]
            if to_list:
                email_sent = _send(
                    to=to_list,
                    subject="MANTHA — Pipeline Report Ready",
                    summary=summary,
                    attachment_paths=[pdf_path],
                )

            return json.dumps({
                "status": "ok",
                "rows_processed": len(df),
                "plots_generated": len(plot_paths),
                "pdf_path": pdf_path,
                "email_sent": email_sent,
                "recipients": to_list,
            })
        except Exception as e:
            return json.dumps({"status": "error", "message": str(e)})

    # ── Tool registry ──────────────────────────────────────────────────────────
    def get_tools(self) -> dict[str, Any]:
        """Return all callable tools for OpenAI function calling."""
        return {
            "fetch_data":         self,
            "transform_data":     self,
            "categorise_data":    self,
            "plot_data":          self,
            "generate_report":    self,
            "send_report":        self,
            "run_full_pipeline":  self,
        }
