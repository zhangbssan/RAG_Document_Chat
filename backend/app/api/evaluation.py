from __future__ import annotations

from fastapi import APIRouter


router = APIRouter()


@router.get("/evaluation")
def evaluation_status() -> dict[str, str]:
    return {"status": "not implemented"}
