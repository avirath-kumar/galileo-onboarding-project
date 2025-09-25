from itertools import product
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict, Optional
import uvicorn
from datetime import datetime
import random
import threading

# Separate fastapi server for product information APIs
api_app = FastAPI(title="Product APIs", version="1.0.0")

# Request / response models
class InventoryRequest(BaseModel):
    product_id: str

class OrderRequest(BaseModel):
    product_id: str
    quantity: int
    customer_email: Optional[str] = "customer@example.com"

class OrderResponse(BaseModel):
    order_id: str
    product_id: str
    product_name: str
    quantity: int
    status: str
    estimated_delivery: str
    total_price: float

# Mock product data
PRODUCTS = {
    "atlas-108": {
        "name": "Aurora Works Atlas 108",
        "inventory": 45,
        "price": 899.99,
        "warehouse": "California"
    },
    "nova-75": {
        "name": "Aurora Works Nova 75",
        "inventory": 12,
        "price": 649.99,
        "warehouse": "Texas"
    },
    "zephyr-87": {
        "name": "Aurora Works Zephyr 87",
        "inventory": 78,
        "price": 749.99,
        "warehouse": "New York"
    }
}

# Thread lock for inventory updates
inventory_lock = threading.Lock()

# Check inventory endpoint
@api_app.get("/inventory/{product_id}")
async def check_inventory(product_id: str):
    # normalize product ID
    product_id = product_id.lower().replace(" ", "-")

    # Map various possible inputs to standard IDs
    if "atlas" in product_id or "108" in product_id:
        product_id = "atlas-108"
    elif "nova" in product_id or "75" in product_id:
        product_id = "nova-75"
    elif "zephyr" in product_id or "87" in product_id:
        product_id = "zephyr-87"
    
    if product_id not in PRODUCTS:
        raise HTTPException(status_code=404, detail=f"Product {product_id} not found")
    
    product = PRODUCTS[product_id]

    # use some variation to simulate real inventory, ensure non negative
    current_stock = product["inventory"] + random.randint(-5, 5)
    current_stock = max(0, current_stock)

    return {
        "product_id": product_id,
        "product_name": product["name"],
        "available_quantity": current_stock,
        "price_per_unit": product["price"],
        "warehouse_location": product["warehouse"],
        "status": "in_stock" if current_stock > 10 else "low_stock" if current_stock > 0 else "out_of_stock"
    }

# Place order endpoint
@api_app.post("/order", response_model=OrderResponse)
async def place_order(order: OrderRequest):
    # Normalize product ID
    product_id = order.product_id.lower().replace(" ", "-")
    
    # Map various possible inputs to standard IDs
    if "atlas" in product_id or "108" in product_id:
        product_id = "atlas-108"
    elif "nova" in product_id or "75" in product_id:
        product_id = "nova-75"
    elif "zephyr" in product_id or "87" in product_id:
        product_id = "zephyr-87"
    
    if product_id not in PRODUCTS:
        raise HTTPException(status_code=404, detail=f"Product {product_id} not found")

    product = PRODUCTS[product_id]

    # Check if enough inventory
    if order.quantity > product["inventory"]:
        raise HTTPException(
            status_code=400,
            detail=f"Insufficient inventory. Only {product['inventory']} units available"
        )
    
    # generate order string / details
    order_id = f"ORD-{datetime.now().strftime('%Y%m%d')}-{random.randint(1000, 9999)}"

    # calculate delivery
    delivery_days = random.randint(3, 5)
    estimated_delivery = f"{delivery_days} business days"

    # Update mock inventory with thread safety
    with inventory_lock:
        PRODUCTS[product_id]["inventory"] -= order.quantity

    # return order response
    return OrderResponse(
        order_id=order_id,
        product_id=product_id,
        product_name=product["name"],
        quantity=order.quantity,
        status="confirmed",
        estimated_delivery=estimated_delivery,
        total_price=round(product["price"] * order.quantity, 2)
    )

# run api on diff port
if __name__ == "__main__":
    uvicorn.run(api_app, host="0.0.0.0", port=8001)