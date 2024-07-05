import streamlit as st
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Dict
import threading
import uvicorn
from fastapi.responses import JSONResponse
from pathlib import Path

# Initialize FastAPI app
api = FastAPI()

# Sample data
data = [
    {"id": 1, "name": "Item 1"},
    {"id": 2, "name": "Item 2"}
]

# Define a Pydantic model for item
class Item(BaseModel):
    id: int
    name: str

# Endpoint to get all items
@api.get('/items', response_model=List[Item])
def get_items():
    return data

# Endpoint to get a specific item by id
@api.get('/items/{item_id}', response_model=Item)
def get_item(item_id: int):
    item = next((item for item in data if item["id"] == item_id), None)
    if item is not None:
        return item
    else:
        return JSONResponse(status_code=404, content={"error": "Item not found"})

# Function to run FastAPI app in a background thread
def run_api():
    uvicorn.run(api, host="0.0.0.0", port=8000)

# Start the FastAPI app in a background thread
threading.Thread(target=run_api, daemon=True).start()

# Streamlit app
def main():
    st.title("Simple API with Streamlit and FastAPI")
    path = Path("logs.txt")
    if not path.exists():
        path.write_text("0")
    count = int(path.read_text())
    count = str(count + 1)
    st.write(f"Count is {count}")
    path.write_text(count)
    st.write("FastAPI is running in the background. You can access the API endpoints at:")
    st.write(" - [Get all items](http://localhost:8000/items)")
    st.write(" - [Get a specific item by ID](http://localhost:8000/items/1)")

if __name__ == '__main__':
    main()
