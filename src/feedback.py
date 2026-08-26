from src.repository.feedback_events import EventTypeEnum, save_feedback_events


def create_feedback_event(
    session_id: str,
    event_type: EventTypeEnum,
    document_id: int | None,
    rank: int | None,
    dwell_time: int | None,
):
    """创建反馈事件"""
    save_feedback_events(session_id, event_type, document_id, rank, dwell_time)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--session-id",
        type=str,
        required=True,
    )
    args = parser.parse_args()
    create_feedback_event(args.session_id, EventTypeEnum.CLICK, 4, 2, None)
