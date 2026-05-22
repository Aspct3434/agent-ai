import { ChatInterface } from './components/ChatInterface';

function App() {
  return (
    <div className="h-screen bg-zinc-950 p-4 sm:p-6">
      <div className="mx-auto flex h-full max-w-5xl flex-col overflow-hidden rounded-2xl border border-zinc-800 bg-zinc-900 shadow-2xl">
        <ChatInterface />
      </div>
    </div>
  );
}

export default App;
