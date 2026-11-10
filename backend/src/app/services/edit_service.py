# 编辑操作服务（V3.0 §12.5：查找替换先 preview 再 apply，批量替换记 edit_operation 支持整组撤销）
from __future__ import annotations

import json

from ..core.database import Database, new_id, now_iso


class EditError(RuntimeError):
    """编辑操作失败（用户可读）。"""


def _latest_run(db: Database, project_id: str) -> str:
    row = db.query_one(
        "SELECT id FROM transcription_runs WHERE project_id=? "
        "AND status='COMPLETED' ORDER BY started_at DESC LIMIT 1", (project_id,))
    if not row:
        raise EditError("该项目尚无完成的转写")
    return row["id"]


def _final_text(row) -> str:
    """final_text = edited ?? original（与 load_segments 一致）。"""
    return row["edited_text"] if row["edited_text"] is not None else row["original_text"]


def preview_replace(db: Database, project_id: str, find: str, replace: str) -> list[dict]:
    """预览全文查找替换：返回受影响片段（不改库）。"""
    if not find:
        raise EditError("查找内容不能为空")
    run_id = _latest_run(db, project_id)
    rows = db.query(
        "SELECT id, sequence_no, start_ms, end_ms, original_text, edited_text "
        "FROM transcript_segments WHERE run_id=? ORDER BY sequence_no", (run_id,))
    hits: list[dict] = []
    for row in rows:
        old = _final_text(row)
        if find in old:
            hits.append({
                "segmentId": row["id"],
                "sequenceNo": row["sequence_no"],
                "startMs": row["start_ms"],
                "endMs": row["end_ms"],
                "oldText": old,
                "newText": old.replace(find, replace),
            })
    return hits


def apply_replace(db: Database, project_id: str, find: str, replace: str) -> dict:
    """执行批量替换：全部命中段写入 edited_text，并记录一个可整组撤销的 edit_operation。

    返回 {operationId, count, segments}。
    """
    hits = preview_replace(db, project_id, find, replace)
    if not hits:
        return {"operationId": None, "count": 0, "segments": []}
    run_id = _latest_run(db, project_id)
    op_id = new_id("edit")
    # payload 保存每段替换前的 edited_text（含 null），undo 时整组恢复
    payload = {
        "find": find,
        "replace": replace,
        "segments": [
            {"id": h["segmentId"], "preEditedText": _pre_edited(db, h["segmentId"])}
            for h in hits
        ],
    }
    for h in hits:
        db.execute(
            "UPDATE transcript_segments SET edited_text=?, updated_at=? WHERE id=?",
            (h["newText"], now_iso(), h["segmentId"]),
        )
    db.execute(
        "INSERT INTO edit_operations (id, project_id, run_id, op_type, payload, created_at) "
        "VALUES (?, ?, ?, 'batch_replace', ?, ?)",
        (op_id, project_id, run_id, json.dumps(payload, ensure_ascii=False), now_iso()),
    )
    return {"operationId": op_id, "count": len(hits), "segments": hits}


def _pre_edited(db: Database, segment_id: str) -> str | None:
    row = db.query_one("SELECT edited_text FROM transcript_segments WHERE id=?", (segment_id,))
    return row["edited_text"] if row else None


def undo_operation(db: Database, operation_id: str) -> dict:
    """整组撤销：恢复 payload 记录的 pre-state。

    替换之后又被手动编辑过的片段（updated_at 晚于操作时间）跳过并报告，
    避免撤销覆盖用户的新编辑。
    """
    op = db.query_one("SELECT * FROM edit_operations WHERE id=?", (operation_id,))
    if not op:
        raise EditError("编辑操作不存在")
    if op["op_type"] != "batch_replace":
        raise EditError(f"操作类型 {op['op_type']} 不支持撤销")

    payload = json.loads(op["payload"])
    restored, skipped = 0, []
    for seg in payload.get("segments", []):
        row = db.query_one(
            "SELECT updated_at FROM transcript_segments WHERE id=?", (seg["id"],))
        if not row:
            skipped.append(seg["id"])
            continue
        if row["updated_at"] > op["created_at"]:
            skipped.append(seg["id"])  # 替换后又有手动编辑，不覆盖
            continue
        db.execute(
            "UPDATE transcript_segments SET edited_text=?, updated_at=? WHERE id=?",
            (seg.get("preEditedText"), now_iso(), seg["id"]),
        )
        restored += 1
    # 记录撤销审计（payload 指向被撤销的操作）
    db.execute(
        "INSERT INTO edit_operations (id, project_id, run_id, op_type, payload, created_at) "
        "VALUES (?, ?, ?, 'undo', ?, ?)",
        (new_id("edit"), op["project_id"], op["run_id"],
         json.dumps({"undone": operation_id}, ensure_ascii=False), now_iso()),
    )
    return {"restored": restored, "skipped": skipped}


def record_single_edit(db: Database, project_id: str, run_id: str,
                       segment_id: str, old: str | None, new: str | None) -> None:
    """单段编辑审计（轻量：不参与撤销，仅记录 from/to）。"""
    db.execute(
        "INSERT INTO edit_operations (id, project_id, run_id, op_type, payload, created_at) "
        "VALUES (?, ?, ?, 'single_edit', ?, ?)",
        (new_id("edit"), project_id, run_id,
         json.dumps({"segmentId": segment_id, "from": old, "to": new}, ensure_ascii=False),
         now_iso()),
    )
