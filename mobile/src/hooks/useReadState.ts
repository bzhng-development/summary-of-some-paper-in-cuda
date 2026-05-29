import { useState, useCallback } from 'react';
import * as db from '../lib/db';

export function useReadState(paperId: string) {
  const [read, setRead] = useState(() => db.isRead(paperId));

  const toggle = useCallback(() => {
    if (read) {
      db.unmarkRead(paperId);
      setRead(false);
    } else {
      db.markRead(paperId);
      setRead(true);
    }
  }, [paperId, read]);

  return { read, toggle };
}
