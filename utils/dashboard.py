from flask import Flask, render_template, jsonify
import threading
from datetime import datetime

# Specify the custom template folder
app = Flask(__name__, template_folder="custom_templates")

dashboard_data = {
    "active_trades": 0,
    "total_profit": 0.0,
    "success_rate": 0.0,
    "live_trades": {"active": 0, "total": 0, "profit": 0.0, "success_rate": 0.0},
    "simulation_trades": {"total": 0, "profit": 0.0, "success_rate": 0.0},
    "trades": [],  # List of recent trades: {"timestamp", "profit", "success", "type"}
    "alerts": [],
}


def update_dashboard_data(trade_data):
    """
    Continuously updates the dashboard data with the latest trade information.

    Args:
        trade_data (list): List of trade data dictionaries.
    """
    while True:
        # Filter live and simulation trades
        live_trades = [t for t in trade_data if t.get("type") == "live"]
        simulation_trades = [t for t in trade_data if t.get("type") == "simulation"]

        # Update live trade stats
        active_live_trades = len([t for t in live_trades if not t.get("completed", True)])
        total_live_trades = len(live_trades)
        live_successful = len([t for t in live_trades if t.get("success", False)])
        live_total_profit = sum(t.get("profit", 0.0) for t in live_trades if t.get("success", False))

        dashboard_data["live_trades"] = {
            "active": active_live_trades,
            "total": total_live_trades,
            "profit": live_total_profit,
            "success_rate": (live_successful / total_live_trades * 100) if total_live_trades else 0,
        }

        # Update simulation trade stats
        total_simulation_trades = len(simulation_trades)
        simulation_successful = len([t for t in simulation_trades if t.get("success", False)])
        simulation_total_profit = sum(t.get("profit", 0.0) for t in simulation_trades if t.get("success", False))

        dashboard_data["simulation_trades"] = {
            "total": total_simulation_trades,
            "profit": simulation_total_profit,
            "success_rate": (simulation_successful / total_simulation_trades * 100) if total_simulation_trades else 0,
        }

        # Update overall stats
        dashboard_data["active_trades"] = active_live_trades
        dashboard_data["total_profit"] = live_total_profit + simulation_total_profit
        total_trades = total_live_trades + total_simulation_trades
        successful_trades = live_successful + simulation_successful
        dashboard_data["success_rate"] = (successful_trades / total_trades * 100) if total_trades else 0

        # Update recent trade history (last 10 trades)
        dashboard_data["trades"] = trade_data[-10:]

        # Generate alerts for negative profit
        dashboard_data["alerts"].clear()
        if dashboard_data["total_profit"] < 0:
            dashboard_data["alerts"].append({
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "message": "Warning: Total profit is negative!",
                "level": "warning",
            })

        # Update every 5 seconds
        threading.Event().wait(5)


@app.route("/api/stats")
def get_stats():
    """
    API endpoint to get the current dashboard statistics.

    Returns:
        Response: JSON response containing the dashboard data.
    """
    return jsonify(dashboard_data)


@app.route("/")
def index():
    """
    Renders the main dashboard page with data.

    Returns:
        str: HTML content of the dashboard page.
    """
    trade_history = {
        "dates": [trade.get("timestamp", "N/A") for trade in dashboard_data["trades"]],
        "profits": [trade.get("profit", 0.0) for trade in dashboard_data["trades"]],
        "types": [trade.get("type", "unknown") for trade in dashboard_data["trades"]],
    }
    return render_template("dashboard.html", data={**dashboard_data, "trade_history": trade_history})


def start_dashboard(trade_data=None):
    """
    Starts the Flask dashboard in a separate thread.

    Args:
        trade_data (list): List of trade data dictionaries (default is an empty list).
    """
    if trade_data is None:
        trade_data = []  # Ensure trade_data is never None

    threading.Thread(target=update_dashboard_data, args=(trade_data,), daemon=True).start()
    app.run(debug=True, port=5000, use_reloader=False)


# Ensure `trade_data` is passed when starting the dashboard
if __name__ == "__main__":
    trade_data = []  # Start with an empty list or load from a file/database
    threading.Thread(target=start_dashboard, args=(trade_data,), daemon=True).start()
