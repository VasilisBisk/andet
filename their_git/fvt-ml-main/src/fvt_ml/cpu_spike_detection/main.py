import typer

from fvt_ml.cpu_spike_detection.std_dev import cli as std_dev_cli

app = typer.Typer()
app.add_typer(std_dev_cli.app, name="stdev")

if __name__ == "__main__":
    app()
