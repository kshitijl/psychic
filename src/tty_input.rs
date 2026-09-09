//! The input thread: terminal bytes in, `Event`s out, stoppable on demand.
//!
//! One thread sits here for the life of the process, asleep in the kernel until
//! the user presses a key. It has to be stoppable, because pressing Enter on a
//! file hands the terminal to `$EDITOR`, and two readers on one terminal means
//! the editor loses keystrokes to us.
//!
//! ## Why this is not just `event::read()` on a thread
//!
//! Crossterm's synchronous API offers exactly two ways to wait for input:
//! `read()`, which blocks forever, and `poll(timeout)`, which blocks for at
//! most the timeout. Neither can be woken by another thread. That is not a bug,
//! it is a missing feature, and every previous version of this file worked
//! around it:
//!
//! * `83ccd32` blocked in `event::read()` and forwarded events into a channel.
//!   A thread parked in `read()` cannot be told to stop, so it kept consuming
//!   the tty while the editor was running and ate the editor's keystrokes.
//! * `9058bec` replaced that with `crossbeam::select!` on the pause channel with
//!   a 10ms `default` arm, then `event::poll(Duration::ZERO)`. Nothing ever
//!   blocked, so pausing worked - at the price of 100 wakeups a second and up to
//!   10ms of latency on every keystroke.
//! * `da7d259` added a 50ms sleep after signalling a pause, hoping the thread
//!   had noticed by then.
//!
//! ## Why a pipe fixes it
//!
//! `crossbeam::select!` waits on *channels*. The terminal is not a channel; it
//! is a file descriptor, and the only thing that can wait on a file descriptor
//! is the kernel. So the two worlds cannot wait on each other:
//!
//! ```text
//!     crossbeam::select!   channels: yes    file descriptors: no
//!     kernel poll(2)       channels: no     file descriptors: yes
//! ```
//!
//! To wait on the terminal *and* on "please stop" in one blocking call, one of
//! them has to cross over. Making the terminal look like a channel is the
//! design that ate keystrokes. So we go the other way and make the stop signal
//! look like a file descriptor: a pipe. [`TtyInput::pause`] writes one byte,
//! and the kernel wakes this thread out of a `poll` that had no timeout at all.
//!
//! That also makes the pause *observable*: the thread answers on an ack channel
//! once it has stopped, so the caller knows the terminal is free instead of
//! sleeping 50ms and hoping.
//!
//! ## What crossterm still does
//!
//! Everything except deciding when to read: escape sequence decoding, the kitty
//! keyboard protocol, SGR mouse, bracketed paste, sequences split across reads,
//! raw mode, the alternate screen, and being ratatui's backend. We call it only
//! with a zero timeout, after the kernel has already said a byte is waiting, so
//! it can never block and the eaten-keystroke bug cannot come back.

use anyhow::{Context, Result};
use crossterm::event::{self, Event};
use std::io::{Read, Write};
use std::os::fd::{AsRawFd, IntoRawFd, RawFd};
use std::os::unix::net::UnixStream;
use std::sync::mpsc::{self, Receiver, RecvTimeoutError, Sender};
use std::time::Duration;

/// How long [`TtyInput::pause`] waits to be told the terminal is free.
///
/// The thread answers as soon as it finishes decoding at most one event, which
/// is microseconds. The bound exists only so that an input thread that has
/// already died cannot stop the user from opening their editor.
const PAUSE_ACK_TIMEOUT: Duration = Duration::from_secs(1);

/// Handle to the input thread.
///
/// Dropping it stops the thread, but prefer [`TtyInput::shutdown`], which also
/// waits for it to finish - see there for why that matters.
pub struct TtyInput {
    /// Write end of the self-pipe. One byte means "stop reading the terminal";
    /// closing it means "shut down".
    wake: UnixStream,
    ack_rx: Receiver<()>,
    resume_tx: Sender<()>,
    handle: Option<std::thread::JoinHandle<()>>,
}

#[cfg(test)]
impl TtyInput {
    /// A handle with no thread behind it, for tests that need an `App` but not
    /// a terminal. Pausing or resuming it does nothing, which is what a test
    /// wants: reading /dev/tty from a test runner is not a thing to do.
    pub fn detached() -> Self {
        let (wake, _dead_end) = std::os::unix::net::UnixStream::pair()
            .expect("a socket pair for a detached input handle");
        let (_ack_tx, ack_rx) = std::sync::mpsc::channel();
        let (resume_tx, _resume_rx) = std::sync::mpsc::channel();
        TtyInput {
            wake,
            ack_rx,
            resume_tx,
            handle: None,
        }
    }
}

/// Proof that the input thread has stopped reading the terminal.
///
/// The thread starts again when this is dropped, so the terminal cannot be left
/// permanently deaf by an early return on the way back from a child process.
pub struct PausedInput<'a> {
    input: &'a TtyInput,
}

impl Drop for PausedInput<'_> {
    fn drop(&mut self) {
        if self.input.resume_tx.send(()).is_err() {
            log::error!("Input thread is gone; it cannot be resumed");
        }
    }
}

impl TtyInput {
    /// Stop reading the terminal, and wait until that has actually happened.
    ///
    /// Call this before handing the terminal to a child process. It returns
    /// once the thread has confirmed it is no longer reading, so the child gets
    /// every keystroke the user types from that moment on.
    pub fn pause(&self) -> PausedInput<'_> {
        // A previous pause that timed out may have left its answer behind.
        // Taking a stale one as this pause's answer would defeat the point.
        while self.ack_rx.try_recv().is_ok() {}

        // The byte itself carries nothing. Writing it is what wakes the thread
        // out of a `poll` with no timeout.
        if let Err(e) = (&self.wake).write_all(&[0]) {
            log::error!("Failed to signal the input thread to pause: {}", e);
        }

        match self.ack_rx.recv_timeout(PAUSE_ACK_TIMEOUT) {
            Ok(()) => {}
            Err(RecvTimeoutError::Timeout) => log::error!(
                "Input thread did not stop within {:?}; running the child anyway, \
                 it may lose keystrokes to us",
                PAUSE_ACK_TIMEOUT
            ),
            Err(RecvTimeoutError::Disconnected) => {
                log::error!("Input thread is gone; nothing is reading the terminal")
            }
        }

        PausedInput { input: self }
    }

    /// Stop the input thread and wait for it to finish.
    ///
    /// Call this before dropping whatever owns the logging channel. The thread
    /// logs on its way out, and it only *starts* going out when this handle is
    /// dropped, so leaving it to `Drop` means it wakes up to find the logger
    /// gone and prints "Error performing logging" over the restored terminal.
    /// The worker thread is shut down explicitly for the same reason.
    ///
    /// Idempotent, and `Drop` calls it, so forgetting is safe if untidy.
    pub fn shutdown(&mut self) {
        // Shutting down the socket rather than dropping it lets this be called
        // twice. Either way, the read end sees end of file, which is the only
        // thing that can reach a thread asleep in the kernel.
        let _ = self.wake.shutdown(std::net::Shutdown::Both);

        // A thread parked waiting to be resumed is not in the kernel and cannot
        // see that, so it would never exit and the join below would hang. That
        // should be impossible - a `PausedInput` guard cannot outlive the call
        // that made it - but hanging on exit is a worse bug than the one this
        // guards against, and the wake-up is one message.
        let _ = self.resume_tx.send(());

        if let Some(handle) = self.handle.take()
            && handle.join().is_err()
        {
            log::error!("Input thread panicked");
        }
    }
}

impl Drop for TtyInput {
    fn drop(&mut self) {
        self.shutdown();
    }
}

/// Start the input thread. Every event it decodes is sent to `event_tx`.
pub fn spawn<T>(event_tx: Sender<T>) -> Result<TtyInput>
where
    T: From<Event> + Send + 'static,
{
    let (wake_tx, wake_rx) = UnixStream::pair().context("Failed to create the input wake pipe")?;
    let (winch_tx, winch_rx) =
        UnixStream::pair().context("Failed to create the resize wake pipe")?;

    // Both read ends are drained until they would block, so they must not block.
    wake_rx
        .set_nonblocking(true)
        .context("Failed to set the wake pipe non-blocking")?;
    winch_rx
        .set_nonblocking(true)
        .context("Failed to set the resize pipe non-blocking")?;

    // A window resize arrives as SIGWINCH, not as bytes on the terminal, so
    // `poll` would sleep straight through it: crossterm has an `Event::Resize`
    // ready, but only hands it over when someone asks, and we only ask when the
    // kernel wakes us. The signal cannot do the waking either, because
    // signal-hook registers with SA_RESTART, so the kernel restarts the
    // interrupted `poll` instead of returning EINTR. Hence a second pipe.
    //
    // Without it a resize is not lost, only late: the 200ms tick redraws, and
    // ratatui re-reads the terminal size on every draw. Measured 134-151ms that
    // way against ~1ms with this pipe - and that fallback goes away as soon as
    // ticks stop firing when nothing is animating.
    //
    // signal-hook keeps a registry of handlers per signal, so ours is added
    // alongside crossterm's rather than replacing it: crossterm still turns the
    // signal into the event, and this only wakes us up to ask for it.
    signal_hook::low_level::pipe::register(signal_hook::consts::SIGWINCH, winch_tx)
        .context("Failed to register the SIGWINCH handler")?;

    let (ack_tx, ack_rx) = mpsc::channel();
    let (resume_tx, resume_rx) = mpsc::channel();

    let (tty, owns_tty) = open_tty()?;

    let handle = std::thread::spawn(move || {
        Reader {
            tty,
            owns_tty,
            wake: wake_rx,
            winch: winch_rx,
            ack_tx,
            resume_rx,
            event_tx,
        }
        .run();
        log::debug!("Input thread exiting");
    });

    Ok(TtyInput {
        wake: wake_tx,
        ack_rx,
        resume_tx,
        handle: Some(handle),
    })
}

/// The descriptor crossterm will read from, so that we wait on the same device.
///
/// Mirrors crossterm's own `tty_fd`, which is not public: stdin when it is a
/// terminal, otherwise `/dev/tty`. Waiting on a *different* descriptor for the
/// same terminal would be fine - readability is a property of the device, not
/// of the descriptor - but it has to be the same device, and stdin is not it
/// when the shell integration has redirected our output.
///
/// Returns the descriptor to wait on, and whether closing it is our job: it is
/// when we opened `/dev/tty` ourselves, and is emphatically not when it is stdin.
fn open_tty() -> Result<(RawFd, bool)> {
    // SAFETY: `isatty` only interrogates a descriptor number and cannot fail
    // in a way that matters here.
    if unsafe { libc::isatty(libc::STDIN_FILENO) } == 1 {
        return Ok((libc::STDIN_FILENO, false));
    }

    let tty = std::fs::OpenOptions::new()
        .read(true)
        .write(true)
        .open("/dev/tty")
        .context("Failed to open /dev/tty to wait for input")?;

    Ok((tty.into_raw_fd(), true))
}

/// Why [`Reader::wait`] returned.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Wake {
    /// Bytes are available on the terminal.
    Terminal,
    /// The window was resized; crossterm has an `Event::Resize` for us.
    Resize,
    /// The UI wants the terminal to itself.
    Pause,
    /// Nothing will ever come again: shut down.
    Closed,
}

/// What draining a self-pipe found.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum PipeState {
    /// Someone wrote to it.
    Signalled,
    /// The write end is gone.
    Closed,
}

struct Reader<T> {
    tty: RawFd,
    /// Whether closing `tty` is our job. False when it is stdin.
    owns_tty: bool,
    wake: UnixStream,
    winch: UnixStream,
    ack_tx: Sender<()>,
    resume_rx: Receiver<()>,
    event_tx: Sender<T>,
}

impl<T> Drop for Reader<T> {
    fn drop(&mut self) {
        if self.owns_tty {
            // SAFETY: `open_tty` handed us this descriptor and nothing else
            // holds it. The thread that polls it is the one being dropped.
            unsafe { libc::close(self.tty) };
        }
    }
}

impl<T> Reader<T>
where
    T: From<Event>,
{
    fn run(mut self) {
        loop {
            // Everything crossterm has already decoded goes out before we sleep
            // again - see `drain_events` for why that matters.
            if !self.drain_events() {
                return;
            }

            match self.wait() {
                // Both mean "ask crossterm what it has now".
                Wake::Terminal | Wake::Resize => {}
                Wake::Pause => {
                    // From here until we are resumed, nothing in this process
                    // reads the terminal. Saying so is what lets the caller
                    // start a child instead of sleeping and hoping.
                    if self.ack_tx.send(()).is_err() {
                        return;
                    }
                    if self.resume_rx.recv().is_err() {
                        return;
                    }
                }
                Wake::Closed => {
                    // A terminal that hangs up reports readable *and* hung up in
                    // the same breath, and the bytes it is holding are as real
                    // as any other. Deliver them before going.
                    self.drain_events();
                    return;
                }
            }
        }
    }

    /// Hand the UI every event crossterm can produce without blocking.
    ///
    /// This has to run to exhaustion before going back to [`Self::wait`].
    /// Crossterm reads the terminal in blocks and can decode several events out
    /// of one read - a paste, or a mouse drag - and those extra events sit in
    /// its queue, not in the kernel's buffer. Waiting on the descriptor while
    /// they are queued would hang until the user happened to press another key.
    ///
    /// Returns false when there is no point continuing.
    fn drain_events(&self) -> bool {
        loop {
            match event::poll(Duration::ZERO) {
                Ok(true) => {}
                Ok(false) => return true,
                Err(e) => {
                    log::error!("Input thread: polling crossterm failed: {}", e);
                    return false;
                }
            }

            match event::read() {
                Ok(event) => {
                    if self.event_tx.send(event.into()).is_err() {
                        // The UI is gone.
                        return false;
                    }
                }
                Err(e) => {
                    log::error!("Input thread: reading an event failed: {}", e);
                    return false;
                }
            }
        }
    }

    /// Sleep until something happens. This is where the thread spends its life.
    fn wait(&mut self) -> Wake {
        const TTY: usize = 0;
        const WAKE: usize = 1;
        const WINCH: usize = 2;
        const NO_TIMEOUT: libc::c_int = -1;

        let mut fds = [
            poll_for(self.tty),
            poll_for(self.wake.as_raw_fd()),
            poll_for(self.winch.as_raw_fd()),
        ];

        loop {
            for fd in fds.iter_mut() {
                fd.revents = 0;
            }

            // SAFETY: `fds` is a live array of three `pollfd`s for the duration
            // of the call, and every descriptor in it is owned by this struct
            // (or is stdin) and stays open across it.
            let ready =
                unsafe { libc::poll(fds.as_mut_ptr(), fds.len() as libc::nfds_t, NO_TIMEOUT) };

            if ready < 0 {
                let err = std::io::Error::last_os_error();
                if err.kind() == std::io::ErrorKind::Interrupted {
                    // A signal was delivered - SIGWINCH is registered, and job
                    // control or a debugger can do this too. Nothing was lost.
                    continue;
                }
                log::error!("Input thread: poll failed: {}", err);
                return Wake::Closed;
            }

            // The pause is checked first on purpose. If the user pressed Enter
            // on a file and typed one more character in the same instant, that
            // character belongs to the editor, not to us.
            if fds[WAKE].revents != 0 {
                return match drain_pipe(&mut self.wake) {
                    PipeState::Signalled => Wake::Pause,
                    PipeState::Closed => Wake::Closed,
                };
            }

            if fds[WINCH].revents != 0 {
                // The write end lives in signal-hook's registry for the life of
                // the process, so this pipe closing means the process is going.
                match drain_pipe(&mut self.winch) {
                    PipeState::Signalled => return Wake::Resize,
                    PipeState::Closed => return Wake::Closed,
                }
            }

            // Hangup before readability, because a terminal that has gone away
            // reports *both*: at end of file the descriptor is permanently
            // readable, so treating that as "there is input" would spin this
            // thread at 100% forever. Nothing is lost by leaving now - the
            // caller drains whatever crossterm still holds before it exits.
            if fds[TTY].revents & (libc::POLLERR | libc::POLLHUP | libc::POLLNVAL) != 0 {
                log::info!("Input thread: the terminal went away");
                return Wake::Closed;
            }
            if fds[TTY].revents & libc::POLLIN != 0 {
                return Wake::Terminal;
            }
        }
    }
}

fn poll_for(fd: RawFd) -> libc::pollfd {
    libc::pollfd {
        fd,
        events: libc::POLLIN,
        revents: 0,
    }
}

/// Empty a self-pipe.
///
/// The bytes carry no information - which pipe woke us is the whole message -
/// so this reads until it would block, and reports whether the write end is
/// still there.
fn drain_pipe(pipe: &mut UnixStream) -> PipeState {
    let mut buffer = [0u8; 64];

    loop {
        match pipe.read(&mut buffer) {
            Ok(0) => return PipeState::Closed,
            Ok(_) => continue,
            Err(e) => {
                return match e.kind() {
                    std::io::ErrorKind::WouldBlock => PipeState::Signalled,
                    std::io::ErrorKind::Interrupted => continue,
                    _ => {
                        log::error!("Input thread: reading a wake pipe failed: {}", e);
                        PipeState::Closed
                    }
                };
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A `Reader` whose "terminal" is a socket pair, so `wait` can be driven
    /// without one. `wait` never touches crossterm, which is the whole reason
    /// it can be tested at all.
    struct Harness {
        reader: Reader<Event>,
        /// Stands in for the terminal: write to it to make the tty readable.
        tty: UnixStream,
        wake: UnixStream,
        winch: UnixStream,
        _ack_rx: Receiver<()>,
        _resume_tx: Sender<()>,
        _event_rx: Receiver<Event>,
    }

    fn harness() -> Harness {
        let (tty_writer, tty_reader) = UnixStream::pair().unwrap();
        let (wake, wake_rx) = UnixStream::pair().unwrap();
        let (winch, winch_rx) = UnixStream::pair().unwrap();
        wake_rx.set_nonblocking(true).unwrap();
        winch_rx.set_nonblocking(true).unwrap();

        let (ack_tx, ack_rx) = mpsc::channel();
        let (resume_tx, resume_rx) = mpsc::channel();
        let (event_tx, event_rx) = mpsc::channel();

        let tty_fd = tty_reader.into_raw_fd();

        Harness {
            reader: Reader {
                tty: tty_fd,
                owns_tty: true,
                wake: wake_rx,
                winch: winch_rx,
                ack_tx,
                resume_rx,
                event_tx,
            },
            tty: tty_writer,
            wake,
            winch,
            _ack_rx: ack_rx,
            _resume_tx: resume_tx,
            _event_rx: event_rx,
        }
    }

    #[test]
    fn test_a_keystroke_wakes_us() {
        let mut h = harness();
        h.tty.write_all(b"x").unwrap();

        assert_eq!(h.reader.wait(), Wake::Terminal);
    }

    #[test]
    fn test_a_pause_wakes_us_out_of_a_wait_with_no_timeout() {
        let mut h = harness();
        h.wake.write_all(&[0]).unwrap();

        assert_eq!(
            h.reader.wait(),
            Wake::Pause,
            "This is the whole point: the thread is blocked in poll with no \
             timeout, and one byte on a pipe gets it out"
        );
    }

    #[test]
    fn test_a_pause_beats_a_keystroke_that_arrived_at_the_same_moment() {
        let mut h = harness();
        h.tty.write_all(b"x").unwrap();
        h.wake.write_all(&[0]).unwrap();

        assert_eq!(
            h.reader.wait(),
            Wake::Pause,
            "The keystroke belongs to whatever the user is about to launch"
        );
    }

    #[test]
    fn test_a_resize_wakes_us() {
        let mut h = harness();
        h.winch.write_all(&[0]).unwrap();

        assert_eq!(
            h.reader.wait(),
            Wake::Resize,
            "A resize arrives as a signal, not as bytes on the terminal, so \
             without its own pipe the UI would not reflow until the next keypress"
        );
    }

    #[test]
    fn test_dropping_the_handle_shuts_the_thread_down() {
        let mut h = harness();
        drop(h.wake);

        assert_eq!(
            h.reader.wait(),
            Wake::Closed,
            "Closing the wake pipe is how shutdown reaches a thread that is \
             asleep in the kernel"
        );
    }

    #[test]
    fn test_a_terminal_that_hangs_up_shuts_the_thread_down() {
        let mut h = harness();
        drop(h.tty);

        assert_eq!(h.reader.wait(), Wake::Closed);
    }

    #[test]
    fn test_a_hung_up_terminal_reports_closed_even_with_bytes_left() {
        let mut h = harness();
        h.tty.write_all(b"x").unwrap();
        drop(h.tty);

        assert_eq!(
            h.reader.wait(),
            Wake::Closed,
            "At end of file the descriptor is readable forever, so reporting \
             Terminal here would spin the thread at 100%. The last keystroke is \
             not lost: `run` drains crossterm once more on its way out"
        );
    }

    #[test]
    fn test_draining_a_pipe_empties_it() {
        let (mut writer, reader) = UnixStream::pair().unwrap();
        reader.set_nonblocking(true).unwrap();
        let mut reader = reader;

        // More than one drain buffer's worth, so the loop has to go round.
        writer.write_all(&[0u8; 200]).unwrap();
        assert_eq!(drain_pipe(&mut reader), PipeState::Signalled);

        // Drained: a second wake would be a spurious one.
        assert_eq!(
            reader.read(&mut [0u8; 8]).unwrap_err().kind(),
            std::io::ErrorKind::WouldBlock
        );
    }

    #[test]
    fn test_draining_reports_a_closed_write_end() {
        let (writer, reader) = UnixStream::pair().unwrap();
        reader.set_nonblocking(true).unwrap();
        let mut reader = reader;
        drop(writer);

        assert_eq!(drain_pipe(&mut reader), PipeState::Closed);
    }
}
