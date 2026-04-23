from lmnr import Laminar, evaluate

# Initialize Laminar for your self-hosted instance
Laminar.initialize(
    project_api_key='Lw6rXeGloI6lhBEIeke6YmtidXrpVBPIe8VVgpburSBs90QMZgq9EDhssc6lD9Hd',
    base_url="http://localhost",
    http_port=8000,
    grpc_port=8001,
)

# 10 Test cases plus your original Canada example
test_data = [
    {"data": {"country": "Canada", "capital": "Ottawa"}, "target": {"capital": "Ottawa"}},
    {"data": {"country": "Singapore", "capital": "Singapore"}, "target": {"capital": "Singapore"}},
    {"data": {"country": "Japan", "capital": "Tokyo"}, "target": {"capital": "Tokyo"}},
    {"data": {"country": "Malaysia", "capital": "Kuala Lumpur"}, "target": {"capital": "Kuala Lumpur"}},
    {"data": {"country": "Laos", "capital": "Vientiane"}, "target": {"capital": "Vientiane"}},
    {"data": {"country": "Thailand", "capital": "Bangtit"}, "target": {"capital": "Bangkok"}},
    {"data": {"country": "Vietnam", "capital": "Saigon"}, "target": {"capital": "Hanoi"}},
    {"data": {"country": "South Korea", "capital": "Seoul"}, "target": {"capital": "Seoul"}},
    {"data": {"country": "France", "capital": "Paris"}, "target": {"capital": "Paris"}},
    {"data": {"country": "Germany", "capital": "Munich"}, "target": {"capital": "Berlin"}},
    {"data": {"country": "Australia", "capital": "Sydney"}, "target": {"capital": "Canberra"}},
]

evaluate(
    data=test_data,
    executor=lambda data: data["capital"],
    evaluators={
        "is_correct": lambda output, target: int(output == target["capital"])
    },
    project_api_key='Lw6rXeGloI6lhBEIeke6YmtidXrpVBPIe8VVgpburSBs90QMZgq9EDhssc6lD9Hd',
    group_name="initial_smoke_test"  # Adding a group name helps organize your dashboard
)