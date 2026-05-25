'use client';

import { useRouter } from 'next/navigation';
import { useEffect, useRef, useState } from 'react';

import { useQueue } from 'hooks/use-queue';

// Watches an end-of-content sentinel. When it enters the viewport, starts a
// 6-second countdown to push the next paper (queue-first, then category-
// adjacent fallback). Cancellable. Mounted at the bottom of the paper.
const AUTO_ADVANCE_MS = 6000;

const AutoAdvance = ({ currentPaperId, fallbackNextHref, fallbackNextTitle }) => {
  const router = useRouter();
  const { list, remove } = useQueue();
  const sentinelRef = useRef(null);
  const [remaining, setRemaining] = useState(null); // ms left or null

  // Pick the next destination: first queued paper (if not the current),
  // else the category-adjacent fallback.
  const queuedNext = list.find((p) => p.id !== currentPaperId) ?? null;
  const targetHref = queuedNext
    ? `/p/${encodeURIComponent(queuedNext.category)}/${encodeURIComponent(queuedNext.slug)}`
    : fallbackNextHref;
  const targetTitle = queuedNext ? queuedNext.title : fallbackNextTitle;

  useEffect(() => {
    if (!targetHref) return undefined;
    const el = sentinelRef.current;
    if (!el) return undefined;

    let timer = null;
    let interval = null;
    let armed = false;

    const cancel = () => {
      if (timer) clearTimeout(timer);
      if (interval) clearInterval(interval);
      timer = interval = null;
      setRemaining(null);
      armed = false;
    };

    const fire = () => {
      cancel();
      if (queuedNext) remove(queuedNext.id);
      router.push(targetHref);
    };

    const arm = () => {
      if (armed) return;
      armed = true;
      const startedAt = Date.now();
      setRemaining(AUTO_ADVANCE_MS);
      interval = setInterval(() => {
        const left = AUTO_ADVANCE_MS - (Date.now() - startedAt);
        if (left <= 0) {
          fire();
        } else {
          setRemaining(left);
        }
      }, 200);
      timer = setTimeout(fire, AUTO_ADVANCE_MS);
    };

    const observer = new IntersectionObserver(
      ([entry]) => {
        if (entry.isIntersecting) arm();
        else cancel();
      },
      { threshold: 0.6 }
    );
    observer.observe(el);

    // Any pointer, key, scroll, or touch activity cancels. Mobile scrolls
    // don't fire `wheel`, so the touch + scroll listeners are mandatory or
    // the countdown will run while the user is still actively reading.
    const onActivity = () => {
      if (armed) cancel();
    };
    window.addEventListener('pointerdown', onActivity);
    window.addEventListener('keydown', onActivity);
    window.addEventListener('wheel', onActivity, { passive: true });
    window.addEventListener('touchstart', onActivity, { passive: true });
    window.addEventListener('touchmove', onActivity, { passive: true });
    window.addEventListener('scroll', onActivity, { passive: true });

    return () => {
      cancel();
      observer.disconnect();
      window.removeEventListener('pointerdown', onActivity);
      window.removeEventListener('keydown', onActivity);
      window.removeEventListener('wheel', onActivity);
      window.removeEventListener('touchstart', onActivity);
      window.removeEventListener('touchmove', onActivity);
      window.removeEventListener('scroll', onActivity);
    };
  }, [router, targetHref, queuedNext, remove]);

  return (
    <>
      <div ref={sentinelRef} className="h-6 w-full" aria-hidden />
      {remaining != null && targetHref ? (
        <div className="pg-no-swipe sticky bottom-[calc(1rem+env(safe-area-inset-bottom))] z-30 mx-auto max-w-md">
          <div className="flex items-center gap-3 rounded-xl border border-primary-1/40 bg-black-new/90 px-4 py-3 shadow-lg backdrop-blur">
            <div className="flex h-10 w-10 shrink-0 items-center justify-center rounded-full border border-primary-1/40 font-mono text-sm text-primary-1 tabular-nums">
              {Math.ceil(remaining / 1000)}
            </div>
            <div className="min-w-0 flex-1">
              <div className="font-mono text-[10px] tracking-wider text-primary-1/80 uppercase">
                {queuedNext ? 'Next in queue' : 'Next in domain'}
              </div>
              <div className="t-sm truncate text-white">{targetTitle}</div>
            </div>
            <button
              type="button"
              onClick={() => {
                // simulate "activity" by dispatching a pointerdown so the
                // effect's cancel path fires too.
                window.dispatchEvent(new Event('pointerdown'));
              }}
              className="rounded-md border border-gray-new-20 px-2.5 py-1 font-mono text-[10px] tracking-wider text-gray-new-70 uppercase transition-colors hover:border-gray-new-30 hover:text-white"
            >
              Cancel
            </button>
          </div>
        </div>
      ) : null}
    </>
  );
};

export default AutoAdvance;
