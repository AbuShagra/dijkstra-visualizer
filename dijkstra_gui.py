import streamlit as st
import networkx as nx
import matplotlib.pyplot as plt
import heapq

st.set_page_config(page_title="Dijkstra Visualizer", layout="centered")
st.title("🔍 Dijkstra's Algorithm Visualizer")

# Initialize session state
if 'edges' not in st.session_state:
    st.session_state.edges = []
if 'directed' not in st.session_state:
    st.session_state.directed = False
if 'graph_updated' not in st.session_state:
    st.session_state.graph_updated = False

# Graph type selector
directed = st.checkbox("Use Directed Graph", value=st.session_state.directed)
st.session_state.directed = directed
G = nx.DiGraph() if directed else nx.Graph()

# Rebuild graph from stored edges
for u, v, w in st.session_state.edges:
    G.add_edge(u, v, weight=w)

# Build graph UI
st.subheader("Build Your Graph")
col1, col2 = st.columns(2)
with col1:
    node1 = st.text_input("From Node", key="from_node")
with col2:
    node2 = st.text_input("To Node", key="to_node")
weight = st.number_input("Edge Weight", min_value=1, value=1, step=1)

if st.button("Add Edge"):
    if node1.strip() and node2.strip():
        st.session_state.edges.append((node1.strip(), node2.strip(), weight))
        st.session_state.graph_updated = True
        st.success(f"Edge added: {node1.strip()} → {node2.strip()} (Weight {weight})")

if st.button("Reset Graph"):
    st.session_state.edges = []
    st.session_state.graph_updated = True
    st.success("Graph has been reset.")

# Show graph only after update
if st.session_state.graph_updated:
    st.subheader("📊 Current Graph")
    pos = nx.spring_layout(G)
    edge_labels = nx.get_edge_attributes(G, 'weight')

    plt.figure(figsize=(6, 4))
    nx.draw(G, pos, with_labels=True, node_color='lightblue', node_size=2000, font_size=14)
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)
    st.pyplot(plt.gcf())

    st.session_state.graph_updated = False

# Run Dijkstra if enough nodes
if G.number_of_nodes() > 1:
    st.subheader("Run Dijkstra")
    start = st.selectbox("Start Node", sorted(G.nodes), key="start_node")
    end = st.selectbox("End Node", sorted(G.nodes), key="end_node")

    def dijkstra_verbose(graph, start):
        distances = {node: float('inf') for node in graph.nodes}
        distances[start] = 0
        previous = {}
        visited = set()
        pq = [(0, start)]
        steps = []

        while pq:
            dist, current = heapq.heappop(pq)
            if current in visited:
                continue
            visited.add(current)
            steps.append(f"Visiting: {current} (Distance: {dist})")

            for neighbor in graph.neighbors(current):
                weight = graph[current][neighbor]['weight']
                new_dist = dist + weight
                steps.append(f"Checking edge {current} → {neighbor} (Weight: {weight})")

                if new_dist < distances[neighbor]:
                    distances[neighbor] = new_dist
                    previous[neighbor] = current
                    heapq.heappush(pq, (new_dist, neighbor))
                    steps.append(f"Updated distance for {neighbor}: {new_dist}")
        return distances, previous, steps

    def get_path(prev, end):
        path = []
        node = end
        while node in prev:
            path.insert(0, node)
            node = prev[node]
        if path:
            path.insert(0, node)
        return path

    if st.button("Find Shortest Path"):
        distances, previous, steps = dijkstra_verbose(G, start)
        path = get_path(previous, end)

        st.subheader("📜 Execution Steps")
        for step in steps:
            st.text(step)

        st.subheader("📌 Results")
        if distances[end] < float('inf'):
            st.success(f"Shortest path from {start} to {end}: {' → '.join(path)}")
            st.info(f"Total cost: {distances[end]}")
        else:
            st.error(f"No path from {start} to {end}")

        st.session_state.graph_updated = True  # Trigger path redraw

        # Show graph with path
        st.subheader("📊 Path Highlight")
        pos = nx.spring_layout(G)
        edge_labels = nx.get_edge_attributes(G, 'weight')

        plt.figure(figsize=(6, 4))
        nx.draw(G, pos, with_labels=True, node_color='lightblue', node_size=2000, font_size=14)
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)
        if len(path) > 1:
            edge_path = [(path[i], path[i+1]) for i in range(len(path)-1)]
            nx.draw_networkx_edges(G, pos, edgelist=edge_path, width=3, edge_color='red')
        st.pyplot(plt.gcf())
