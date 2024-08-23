import AnomalyDetection from './AnomalyDetection';
import ModelDocumentation from './ModelDocumentation';

export default function Home() {
    return (
        <main className="main-content">
            <AnomalyDetection />
            <ModelDocumentation />
        </main>
    );
}
