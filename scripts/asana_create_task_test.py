import os

from asana import ApiClient, Configuration
from asana.rest import ApiException
from asana.api import TasksApi, WorkspacesApi, ProjectsApi
from dotenv import load_dotenv

load_dotenv()

config = Configuration()

# Load PAT from environment variable
config.access_token = os.environ.get('ASANA_PAT')

client = ApiClient(config)

# projects_api = ProjectsApi(client)
# workspaces_api = WorkspacesApi(client)
# workspaces = workspaces_api.get_workspaces(opts={})
# for w in workspaces:
#     print("  ", w['gid'], w['name'])

#     projects = projects_api.get_projects(opts={"workspace": w['gid']})
#     for p in projects:
#         print("  ", p['gid'], p['name'])

# Workspace & Project IDs
workspace_id = os.environ.get("ASANA_WORKSPACE_ID")
project_id = os.environ.get("ASANA_PROJECT_ID")

tasks_api = TasksApi(client)

try:
    body = {
        "data": {
            "name": "New Task (v5)",
            "notes": "Created with dict body",
            "workspace": workspace_id,
            "projects": [project_id]
        }
    }
    opts = {}
    new_task = tasks_api.create_task(body=body, opts=opts)
    print("Task created:", new_task['gid'])

except ApiException as e:
    print("Error:", e)
