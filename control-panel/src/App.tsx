import { ChatInterface } from './components/ChatInterface';

function App() {
  return (
    <div className="h-screen bg-zinc-950 p-3 sm:p-5">
      <div className="mx-auto flex h-full max-w-7xl flex-col overflow-hidden rounded-lg border border-zinc-800 bg-zinc-900 shadow-2xl">
        <ChatInterface />
      </div>
    </div>
  );
}

export default App;
