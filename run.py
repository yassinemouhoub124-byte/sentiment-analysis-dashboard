import os
import sys
import streamlit.web.cli as stcli

if __name__ == "__main__":
    # Point directly to app.py location
    project_root = os.path.dirname(os.path.abspath(__file__))
    app_path = os.path.join(project_root, "app.py")

    sys.argv = ["streamlit", "run", app_path]
    sys.exit(stcli.main())